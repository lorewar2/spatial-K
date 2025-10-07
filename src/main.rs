#![allow(dead_code)]
use core::f32;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Poisson, Distribution, Gamma};

const DATA_FILE: &'static str = "./data/data.csv";
const DATA_FILE_2: &'static str = "./data/dense.csv";
const ANNO_FILE: &'static str = "./data/NatGen2022_scRNAseq_annotations.csv";
const OUT_FILE: &'static str = "./result.tsv";
const K: usize = 6;
const GENE_NUM_SIM: usize = 10;
const CELL_NUM_SIM: usize = 30;
const SEED: u64 = 2;

fn main() {
    let all_cell_data_original = data_loader_scrna(SEED);
    //let (all_cell_data_original, cell_ids) = data_loader_spatial();
    let best_total_loss = -f32::INFINITY;
    let mut _best_final_log_loss;
    let alpha_vec = vec![1.0, 2.0 , 3.0, 4.0, 5.0, 6.0];
    for seed in 0..6 {
        println!("SEED {}", seed);
        let all_cell_data = all_cell_data_original.clone();
        let gene_count = all_cell_data[0].gene_count;
        // initialize cluster centers
        //let mut cluster_centers = init_cluster_centers_uniform(gene_count, K, seed);
        let mut cluster_centers = init_cluster_centers_gamma(gene_count, K, &all_cell_data, alpha_vec[seed as usize], &0);
        //let mut cluster_centers = init_cluster_centers_optimal(gene_count, K, &all_cell_data);
        let mut cluster_weights = vec![(1.0 / (K as f32)).ln(); K];
        // run EM
        let (log_loss_final, total_loss) = em(gene_count, &mut cluster_centers, &mut cluster_weights, &all_cell_data, 1020.0);
        let mut string_vec = vec![];
        for cell in all_cell_data {
            string_vec.push(cell.cell_type);
        }
        if total_loss > best_total_loss {
            _best_final_log_loss = log_loss_final;
        }
    }
    //data_writer(cell_ids, _best_final_log_loss);
}

fn em(gene_count: usize, mut cluster_centers: &mut Vec<Vec<f32>>, mut cluster_weights: &mut Vec<f32>, all_cell_data: &Vec<CellData>, temp: f32) -> (Vec<Vec<f32>>, f32) {
    // vec to save the log loss for each cluster from each cell, to determine the one with least
    let mut log_loss_final = vec![vec![]; all_cell_data.len()];
    let num_clusters = K;
    let cell_count = all_cell_data.len();
    let mut last_log_loss = 0.0;
    // update probs to update the cc
    let mut update_prob: Vec<Vec<f32>> = Vec::new();
    let mut update_weight: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..num_clusters {
        update_prob.push(Vec::new());
        update_weight.push(Vec::new());
        for _index in 0..gene_count {
            update_prob[cluster].push(0.0001);
            update_weight[cluster].push(0.0);
        }
    }
    // run 10 times and see
    for run in 0..10 {
        let mut log_poisson_total = 0.0;
        // reset
        reset_update_prob(num_clusters, gene_count, &mut update_prob);
        for (celldex, cell) in all_cell_data.iter().enumerate() {
            // calculate poisson loss here // Modify with cluster weights
            let log_poisson = poisson_loss(cell, &cluster_centers, &cluster_weights);
            log_loss_final[celldex] = log_poisson.clone();
            // sum up the total loss
            log_poisson_total += log_sum_exp(&log_poisson);
            // update the temp probs
            update_update_prob(&mut update_prob, &mut update_weight, cell, &log_poisson, cell_count, temp);
        }
        //println!("BEFORE UPDATE CC {:?}", cluster_centers);
        update_cluster_centers(gene_count, &update_prob, &mut cluster_centers);
        // update the cluster weights, without this should be same as const prior
        //update_cluster_weights(gene_count, &update_weight, &mut cluster_weights);
        // display stuff
        let log_loss_change = log_poisson_total - last_log_loss;
        last_log_loss = log_poisson_total;
        println!("\nPoisson\trun:{}\tloss:{}\tchange:{}\n", run, log_poisson_total, log_loss_change);
        // check the cluster assignment
        let mut assigned_vec: Vec<usize> = vec![0; num_clusters];
        let mut cluster_test = vec![vec![]; num_clusters];
        let mut ground_truth = vec![];
        let mut predicted = vec![];
        let type_vec = vec!["Epithelial", "Fibroblast", "Endothelial", "Endocrine", "Immune", "Schwann"];
        // Print stuff for testing
        for (index, final_log_probability) in log_loss_final.iter().enumerate() {
            let index_of_max: usize = final_log_probability.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
            assigned_vec[index_of_max] += 1;
            cluster_test[index_of_max].push(all_cell_data[index].cell_type.clone());
            predicted.push(index_of_max);
            for (index2, _type) in type_vec.iter().enumerate() {
                if *_type == all_cell_data[index].cell_type.clone() {
                    ground_truth.push(index2);
                }
            }
        }
        let rand_index = rand_index_calculator(&predicted, &ground_truth);
        println!("Rand Index {}\n", rand_index);
        for (index, cluster) in cluster_test.iter().enumerate() {
            println!("Cluster {}\tweight:{}\tcells:{} ", index, cluster_weights[index].exp(), assigned_vec[index]);
            let mut counts = HashMap::new();
            for item in cluster {
                *counts.entry(item).or_insert(0) += 1;
            }
            for (key, value) in &counts {
                print!("{}: {}\t", key, value);
            }
            println!("\n");
        }
        //println!("AFTER UPDATE CC {:?}", cluster_centers);
    }
    (log_loss_final, last_log_loss)
}

fn data_loader_scrna(seed: u64) -> Vec<CellData> {
    println!("Start Data Loading");
    let mut rng = StdRng::seed_from_u64(seed);
    let cells_to_load = 2000;
    let equal_cell = true;

    // annotation loading for the cells (cell types)
    let mut cell_types = vec![];
    let mut required_indices = vec![];
    let mut required_cell_types = vec![];
    let mut cell_type_4_indices: Vec<String> = vec![];
    let mut indices_4_cell_type: Vec<Vec<usize>> = vec![];
    let mut total_cells = 0;

    let anno_file = File::open(ANNO_FILE).expect("cannot open data file");
    let anno_data_reader = BufReader::new(&anno_file);
    for (line_index, line) in anno_data_reader.lines().enumerate().skip(1) {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        cell_type_4_indices.push(values[96].to_string());
        match cell_types.iter().position(|name| name == &values[96].to_string()) {
            Some(x) => {
                indices_4_cell_type[x].push(line_index - 1);
            }
            None => {
                if values[96].to_string() != "unknown" {
                    cell_types.push(values[96].to_string());
                    indices_4_cell_type.push(vec![line_index - 1]);
                }
            }
        }
        total_cells += 1;
    }
    // select 1000 / cell types from each cell type
    if equal_cell {
        let required_per_type = cells_to_load / cell_types.len();
        for (index, cell_type) in cell_types.iter().enumerate() {
            for _ in 0..required_per_type {
                let select_index = rng.gen_range(0..indices_4_cell_type[index].len());
                required_indices.push(indices_4_cell_type[index][select_index]);
                required_cell_types.push(cell_type.clone());
            }
        }
    }
    // randomly select 1000 cells
    else {
        // select the required cells randomly from 0..cells_to_load
        for _ in 0..cells_to_load {
            // 1 (inclusive) to 21 (exclusive)
            let select_index = rng.gen_range(0..total_cells);
            required_indices.push(select_index);
            required_cell_types.push(cell_type_4_indices[select_index].clone());
        }
    }
    //println!("{:?} {:?}", required_indices, required_cell_types);
    // load the required data
    let mut all_cell_data= vec![];
    let data_file = File::open(DATA_FILE_2).expect("cannot open data file");
    let data_reader = BufReader::new(&data_file);
    for (line_index, line) in data_reader.lines().enumerate() {
        match required_indices.iter().position(|name| name == &line_index) {
            Some(x) => {
                let line = line.unwrap();
                let values: Vec<&str> = line.split(',').collect();
                let mut cell_data = CellData::new(values.len());
                cell_data.cell_type = required_cell_types[x].clone();
                for (gene_index, value) in values.iter().enumerate() {
                    // convert to u32 and add to cell data
                    let read_count = (value.to_string().parse::<f32>().unwrap() * 10.0) as u16;
                    cell_data.read_counts[gene_index] = read_count;
                }
                all_cell_data.push(cell_data);
            }
            None => {}
        }
    }
    println!("{}", all_cell_data.len());
    // load the gene list
    println!("{}", all_cell_data[0].gene_count);

    println!("End Data Loading");
    all_cell_data
}

fn poisson_loss(cell: &CellData, cluster_centers: &Vec<Vec<f32>>, log_cluster_weight: &Vec<f32>) -> Vec<f32> {
    let mut log_probabilities: Vec<f32> = Vec::new();
    for (cluster, center) in cluster_centers.iter().enumerate() {
        log_probabilities.push(log_cluster_weight[cluster]);
        for (gene_index, read_count) in cell.read_counts.iter().enumerate() {
            //log(P)=-λ+x​log(λ​)-log(x​!)
            //x = read count 
            //λ = center[gene_index]
            let mut mod_read_count = *read_count;
            if mod_read_count > 100 {
                mod_read_count = 100;
            }
            let value = - center[gene_index] + mod_read_count as f32 * center[gene_index].ln();
            let factorial = (factorial(mod_read_count) as f32 + 0.000000001).ln();
            log_probabilities[cluster] += value - factorial;
        }
    }
    log_probabilities
}

fn update_cluster_weights(gene_count: usize, update_weight: &Vec<Vec<f32>>, cluster_weights: &mut Vec<f32>) {
    for cluster in 0..update_weight.len() {
        let mut total_cluster_update = 0.0;
        for locus in 0..gene_count {
            let update = update_weight[cluster][locus];
            total_cluster_update += update;
        }
        cluster_weights[cluster] = total_cluster_update;
    }
    let sum: f32 = cluster_weights.iter().sum();
    let normalized: Vec<f32> = cluster_weights.iter().map(|&x| (x / sum).ln()).collect();
    *cluster_weights = normalized;
}

fn update_cluster_centers(gene_count: usize, update_prob: &Vec<Vec<f32>>, cluster_centers: &mut Vec<Vec<f32>>) {
    for locus in 0..gene_count {
        for cluster in 0..update_prob.len() {
            let update = update_prob[cluster][locus];
            cluster_centers[cluster][locus] = update.max(0.0).min(9.99);
        }
    }
}

fn update_update_prob(update_prob: &mut Vec<Vec<f32>>, update_weight: &mut Vec<Vec<f32>>, cell: &CellData, log_poisson: &Vec<f32>, cell_count: usize, temp_step: f32) {
    for locus in 0..cell.gene_count {
        // get the sum
        let sum = log_sum_exp(log_poisson);
        for (cluster, probability) in log_poisson.iter().enumerate() {
            // normalize and turn log to normal
            let update_prob_exp = ((0.0000001 + probability - sum) / temp_step).exp() / (cell_count as f32 / 2.9); // 2.9 best
            update_prob[cluster][locus] += update_prob_exp * (cell.read_counts[locus] as f32);
            update_weight[cluster][locus] += update_prob_exp;
        }
    }
}

fn factorial (n: u16) -> u64 {
    let n = n as u64;
    let mut result = 1;
    for i in 1..=n {
        result *= i;
    }
    result
}

fn log_sum_exp(p: &Vec<f32>) -> f32{
    let max_p: f32 = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_rst: f32 = p.iter().map(|x| (x - max_p).exp()).sum();
    max_p + sum_rst.ln()
}

fn reset_update_prob(num_clusters: usize, gene_count: usize, update_prob: &mut Vec<Vec<f32>>) {
    for cluster in 0..num_clusters {
        for index in 0..gene_count {
            update_prob[cluster][index] = 0.0001;
        }
    }
}

fn init_cluster_centers_uniform(gene_count: usize, num_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed as u64);
    // Generate random numbers using the seeded RNG
    let mut centers: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..num_clusters {
        centers.push(Vec::new());
        for _ in 0..gene_count {
            centers[cluster].push((rng.gen::<f32>() * 3.0).min(3.0).max(0.0001));
        }
    }
    centers
}

fn init_cluster_centers_optimal(gene_count: usize, cluster_num: usize, all_cell_data: &Vec<CellData>) -> Vec<Vec<f32>> {
    let mut sum_for_cell_type = vec![vec![0; gene_count]; cluster_num];
    let mut count_for_cell_type = vec![vec![0; gene_count]; cluster_num];
    let mut mean_for_cell_type = vec![vec![0.0; gene_count]; cluster_num];
    // Should make this better
    for cell_data in all_cell_data {
        let cell_type = cell_data.cell_type.clone();
        let mut cluster = 2;
        if cell_type == "Epithelial" {
            cluster = 0;
        }
        else if cell_type == "Fibroblast" {
            cluster = 1;
        }
        else if cell_type == "Endothelial" {
            cluster = 2;
        }
        else if cell_type == "Endocrine" {
            cluster = 3;
        }
        else if cell_type == "Immune" {
            cluster = 4;
        }
        else if cell_type == "Schwann" {
            cluster = 5;
        }
        if cluster >= cluster_num {
            cluster = cluster_num - 1;
        }
        for (index, value) in cell_data.read_counts.iter().enumerate() {
            if *value != 0 {
                sum_for_cell_type[cluster][index] += value;
                count_for_cell_type[cluster][index] += 1;
            }
        }
    }
    for cluster in 0..cluster_num {
        for gene_index in 0..gene_count {
            mean_for_cell_type[cluster][gene_index] = (sum_for_cell_type[cluster][gene_index] as f32 + 0.000000001) / (6.9 * count_for_cell_type[cluster][gene_index] as f32 + 0.0001);
        }
    }
    mean_for_cell_type
}

fn init_cluster_centers_gamma(gene_count: usize, num_clusters: usize, all_cell_data: &Vec<CellData>, alpha: f32, seed: &u64) -> Vec<Vec<f32>> {
    // initialize values
    let mut centers: Vec<Vec<f32>> = vec![vec![]; num_clusters];
    let mut rng = StdRng::seed_from_u64(*seed);
    let mut read_counts_gene = vec![vec![]; gene_count];
    let mut non_zero_counts = vec![0; gene_count];
    let mut read_counts_gene_sum = vec![0; gene_count];
    let mut means = vec![];
    // count the non zeros
    for cell_data in all_cell_data {
        for (index, value) in cell_data.read_counts.iter().enumerate() {
            if value != &0 {
                read_counts_gene[index].push(*value as usize);
                read_counts_gene_sum[index] += *value as usize;
                non_zero_counts[index] += 1;
            }
        }
    }
    // calculate mean per gene
    for (_index, read_count) in read_counts_gene.iter().enumerate() {
        if read_count.len() > 0 {
            let mut temp = read_count.clone();
            temp.sort();
            means.push(temp[temp.len() / 4] as f32);
        }
        else {
            means.push(0.0000001);
        }
    }
    // using means in gamma draw values for clusters
    println!("alpha {}", alpha);
    for mean in means {
        let theta = mean / alpha;
        let gamma = Gamma::new(alpha, theta).expect("invalid");
        for cluster in 0..num_clusters {
            let drawn_value = gamma.sample(&mut rng);
            centers[cluster].push(drawn_value);
        }
    }
    centers
}

fn data_loader_spatial() -> (Vec<CellData>, Vec<String>) {
    println!("Start Data Loading");
    let data_file = File::open(DATA_FILE).expect("cannot open data file");
    let data_reader = BufReader::new(data_file);
    let mut all_cell_data= vec![];
    let mut cell_ids = vec![];
    for (line_index, line) in data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        if line_index == 0 {
            // this is the header, make new cell vector
            all_cell_data = vec![CellData::new(0); values.len()];
            // save the values in a vector, cell id_fov_etc
            for value in values {
                cell_ids.push(value.trim().to_string());
            }
            continue;
        }
        let mut gene_expressed_by_cells = 0;
        for (_cell_index, value) in values.iter().enumerate() {
            // convert to u32 and add to cell data
            let read_count = value.to_string().parse::<u16>().unwrap();
            if read_count > 1 {
                gene_expressed_by_cells += 1;
            }
        }
        if gene_expressed_by_cells > 3000 {
            println!("gene {} passed", line_index);
            for (cell_index, value) in values.iter().enumerate() {
                // convert to u32 and add to cell data
                let read_count = value.to_string().parse::<u16>().unwrap();
                all_cell_data[cell_index].read_counts.push(read_count);
                all_cell_data[cell_index].gene_count = all_cell_data[cell_index].gene_count + 1;
            }
        }  
    }
    println!("End Data Loading");
    (all_cell_data, cell_ids)
}

fn data_writer(cell_ids: Vec<String>, log_loss_final: Vec<Vec<f32>>) {
    // write to file
    let mut file = File::create(OUT_FILE).unwrap();
    for (cell_id, log_loss) in cell_ids.iter().zip(log_loss_final.iter()) {
        let index_of_max: usize = log_loss.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
        writeln!(file, "{}\t{}\t{:?}", cell_id, index_of_max, log_loss).expect("result file cannot be written");
    }
}

fn data_simulator (seed: u64) -> (Vec<CellData>, Vec<usize>) {
    // initializations
    let number_of_genes = GENE_NUM_SIM;
    let number_of_cells = CELL_NUM_SIM;
    let number_of_clusters = K;
    let gene_non_zero_prob = 0.2;
    // random assignment of lambda for cluster for gene
    let mut lambda_vec: Vec<Vec<usize>> = vec![vec![]; number_of_clusters];
    // convert this to all cell data
    let mut data_vec: Vec<CellData> = vec![CellData::new(number_of_genes); number_of_cells];
    let mut cluster_assignment: Vec<usize> = vec![];
    let mut gene_non_zero_cluster = vec![];
    let mut rng = StdRng::seed_from_u64(seed);
    // assign a 90 % of the genes of a cluster 0 reads
    for _cluster in 0..number_of_clusters {
        let mut gene_non_zero = vec![];
        for gene in 0..number_of_genes {
            if rng.gen_bool(gene_non_zero_prob) {
                gene_non_zero.push(gene);
            }
        }
        gene_non_zero_cluster.push(gene_non_zero);
    }
    // assign a random cluster for each cell
    for _cell in 0..number_of_cells {
        cluster_assignment.push(rng.gen_range(0..number_of_clusters));
    }
    for cluster in 0..number_of_clusters {
        for gene in 0..number_of_genes {
            let generated_lamda = rng.gen_range(1..50);
            lambda_vec[cluster].push(generated_lamda);
            // draw from each distribution and populate the celldata
            let poisson = Poisson::new(generated_lamda as f32).unwrap();
            if gene_non_zero_cluster[cluster].contains(&gene) {

            }
            else {
                continue;
            }
            for cell in 0..number_of_cells {
                // if this cluster generate data for gene
                if cluster_assignment[cell] == cluster {
                    let sample_data: f32 = poisson.sample(&mut rng);
                    data_vec[cell].read_counts[gene] = sample_data as u16;
                }   
            }
        }
    }
    // print stuff
    for cluster in 0..number_of_clusters {
        println!("!!!!!!!! CLUSTER  {}", cluster);
        println!("lamda vector {:?}", lambda_vec[cluster]);
        for cell in 0..number_of_cells {
            if cluster_assignment[cell] == cluster {
                print!("cell number {}", cell);
                print!(" {:?}", data_vec[cell].read_counts);
                println!("");
            }
        }
    }
    (data_vec, cluster_assignment)
}


fn result_display(log_loss_final: &Vec<Vec<f32>>, ground_truth: &Vec<usize>) {
    let mut method_assignment = vec![];
    for (_index, final_log_probability) in log_loss_final.iter().enumerate() {
        let index_of_max: usize = final_log_probability.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
        //println!("cell {} assigned {}", index, index_of_max);
        method_assignment.push(index_of_max);
    }
    //sleep(Duration::from_secs(2));
    let rand_index = rand_index_calculator(&method_assignment, ground_truth);
    println!("Rand Index {}", rand_index);
}

fn rand_index_calculator (predicted: &Vec<usize>, ground_truth: &Vec<usize>) -> f64 {
    assert_eq!(ground_truth.len(), predicted.len());
    let n = ground_truth.len();

    let mut agree = 0usize;
    let mut total = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            let same_truth = ground_truth[i] == ground_truth[j];
            let same_pred = predicted[i] == predicted[j];

            if same_truth == same_pred {
                agree += 1;
            }
            total += 1;
        }
    }
    agree as f64 / total as f64
}

#[derive(Clone)]
struct CellData {
    read_counts: Vec<u16>,
    gene_count: usize,
    cell_type: String
}
impl CellData {
    fn new(gene_count: usize) -> CellData {
        CellData{
            read_counts: vec![0; gene_count],
            gene_count: gene_count,
            cell_type: String::new()
        }
    }
}