use std::fs::File;
use std::io::{BufRead, BufReader, Seek, Write};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Poisson, Distribution, Gamma};

const DATA_FILE: &'static str = "./data/data.csv";
const DATA_FILE_2: &'static str = "./data/dense.csv";
const OUT_FILE: &'static str = "./result.tsv";
const K: usize = 3;
const GENE_NUM_SIM: usize = 10;
const CELL_NUM_SIM: usize = 30;
const SEED: u64 = 25;

fn main() {
    data_loader_scrna();
    //let (_all_cell_data, _ground_truth) = data_simulator(SEED);
    // let (all_cell_data, cell_ids) = data_loader_spatial();
    // let seed = SEED;
    // let alpha = 4.5;
    // let gene_count = all_cell_data[0].gene_count;
    // // initialize the cluster centers gamma
    // //let mut cluster_centers = init_cluster_centers_uniform(gene_count, K, seed);
    // let mut cluster_centers = init_cluster_centers_gamma(gene_count, K, &all_cell_data, alpha, seed);
    // let log_loss_final = em(gene_count, &mut cluster_centers, &all_cell_data);
    // //result_display(&log_loss_final, &ground_truth);
    // data_writer(cell_ids, log_loss_final);
}

fn em(gene_count: usize, mut cluster_centers: &mut Vec<Vec<f32>>, all_cell_data: &Vec<CellData>) -> Vec<Vec<f32>> {
    println!("Starting cc {:?}", cluster_centers);
    // vec to save the log loss for each cluster from each cell, to determine the one with least
    let mut log_loss_final = vec![vec![]; all_cell_data.len()];
    let num_clusters = K;
    let cell_count = all_cell_data.len();
    // log prior assume all clusters are equal
    let log_prior: f32 = (1.0 / (num_clusters as f32)).ln();
    let mut last_log_loss = 0.0;
    // update probs to update the cc
    let mut update_prob: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..num_clusters {
        update_prob.push(Vec::new());
        for _index in 0..gene_count {
            update_prob[cluster].push(0.000000001);
        }
    }
    // run 10 times and see
    for run in 0..30 {
        let mut log_poisson_total = 0.0;
        // reset
        reset_update_prob(num_clusters, gene_count, &mut update_prob);
        for (celldex, cell) in all_cell_data.iter().enumerate() {
            // calculate poisson loss here
            let log_poisson = poisson_loss(cell, &cluster_centers, log_prior);
            log_loss_final[celldex] = log_poisson.clone();
            // sum up the total loss
            log_poisson_total += log_sum_exp(&log_poisson);
            // update the temp probs
            update_update_prob(&mut update_prob, cell, &log_poisson, cell_count);
        }
        //println!("BEFORE UPDATE CC {:?}", cluster_centers);
        update_cluster_centers(gene_count, &update_prob, &mut cluster_centers);
        // display stuff
        let log_loss_change = log_poisson_total - last_log_loss;
        last_log_loss = log_poisson_total;
        println!("poisson\t{}\t{}\t{}", run, log_poisson_total, log_loss_change);
        // check the cluster assignment
        let mut assigned_vec: Vec<usize> = vec![0; num_clusters];
        for (_index, final_log_probability) in log_loss_final.iter().enumerate() {
            let index_of_max: usize = final_log_probability.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
            assigned_vec[index_of_max] += 1;
            //println!("cell {} assigned {}", index, index_of_max);
        }
        //println!("AFTER UPDATE CC {:?}", cluster_centers);
        println!("Assignment vec {:?}\n", assigned_vec);
    }
    log_loss_final
}

fn update_cluster_centers(gene_count: usize, update_prob: &Vec<Vec<f32>>, cluster_centers: &mut Vec<Vec<f32>>) {
    for locus in 0..gene_count {
        for cluster in 0..update_prob.len() {
            let update = update_prob[cluster][locus];
            //println!("{}", update);
            cluster_centers[cluster][locus] = update;
        }
    }
}

fn update_update_prob(update_prob: &mut Vec<Vec<f32>>, cell: &CellData, log_poisson: &Vec<f32>, cell_count: usize) {
    for locus in 0..cell.gene_count {
        // get the sum
        let sum = log_sum_exp(log_poisson);
        for (cluster, probability) in log_poisson.iter().enumerate() {
            // normalize and turn log to normal
            let update_prob_exp = (probability - sum).exp() / (cell_count as f32 / 2.9);
            //println!("{}", update_prob_exp);
            update_prob[cluster][locus] += update_prob_exp * (cell.read_counts[locus] as f32);
        }
    }
}

fn poisson_loss(cell: &CellData, cluster_centers: &Vec<Vec<f32>>, log_prior: f32) -> Vec<f32> {
    let mut log_probabilities: Vec<f32> = Vec::new();
    for (cluster, center) in cluster_centers.iter().enumerate() {
        log_probabilities.push(log_prior);
        for (gene_index, read_count) in cell.read_counts.iter().enumerate() {
            //log(P)=-λ+x​log(λ​)-log(x​!)
            //x = read count 
            //λ = center[gene_index]
            let mut mod_read_count = *read_count;

            if mod_read_count > 100 {
                mod_read_count = 100;
            }
            //println!("read {}", mod_read_count);
            //println!("fact {} others {}", (factorial(mod_read_count) as f32).ln(), - center[gene_index] + mod_read_count as f32 * center[gene_index].ln());
            let value = - center[gene_index] + mod_read_count as f32 * center[gene_index].ln();
            let factorial = (factorial(mod_read_count) as f32 + 0.000000001).ln();
            log_probabilities[cluster] += value - factorial;
        }
    }
    log_probabilities
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
            centers[cluster].push((rng.gen::<f32>() * 10.0).min(10.0).max(0.0001));
        }
    }
    centers
}

fn init_cluster_centers_gamma(gene_count: usize, num_clusters: usize, all_cell_data: &Vec<CellData>, alpha: f32, seed: u64) -> Vec<Vec<f32>> {
    // initialize values
    let mut centers: Vec<Vec<f32>> = vec![vec![]; num_clusters];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut read_counts_gene = vec![vec![]; gene_count];
    //let mut total_cells = all_cell_data.len();
    let mut non_zero_counts = vec![0; gene_count];
    let mut read_counts_gene_sum = vec![0; gene_count];
    let mut means = vec![];
    //println!("{}", cell_count);
    for cell_data in all_cell_data {
        for (index, value) in cell_data.read_counts.iter().enumerate() {
            if value != &0 {
                read_counts_gene[index].push(*value as usize);
                read_counts_gene_sum[index] += *value as usize;
                non_zero_counts[index] += 1;
            }
        }
    }
    //println!("{:?}", read_counts_gene);
    // calculate mean per gene
    for (_index, read_count) in read_counts_gene.iter().enumerate() {
        if read_count.len() > 0 {
            let mut temp = read_count.clone();
            temp.sort();
            // println!("Gene num: {}", index);
            // println!("Total_entries: {} Non_zero_entries: {} Zero_entries: {} ", total_cells, temp.len(), total_cells - temp.len());
            // println!("Max: {} Min: {} Median: {} Mean (Non zero): {} Mean (Zero): {}", temp[temp.len() - 1], temp[0], temp[temp.len() / 2], read_counts_gene_sum[index] as f32 / non_zero_counts[index] as f32, read_counts_gene_sum[index] as f32 / total_cells as f32);
            // println!("top: {} {} {} {} bottom (non zero): {} {} {} {}", temp[temp.len() - 1], temp[temp.len() - 2], temp[temp.len() - 3], temp[temp.len() - 4], temp[0], temp[1], temp[2], temp[3]);
            means.push(temp[temp.len() / 2] as f32);
        }
        else {
            means.push(0.0000001);
        }
    }
    //println!("{:?}", means);
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
    //println!("{:?}", centers);
    centers
    //vec![vec![11.0, 24.0], vec![43.0, 25.0]]
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
    // sort the all data based on one gene entry just to see
    // let mut all_sorted = vec![];
    // let mut indices: Vec<usize> = (0..all_cell_data[0].read_counts.len()).collect();
    
    // indices.sort_by_key(|&i| all_cell_data[0].read_counts[i]);
    // indices.reverse();
    // for index in 0..all_cell_data.len() {
    //     let sorted: Vec<_> = indices.iter().map(|&i| all_cell_data[index].read_counts[i]).collect();
    //     if sorted[0] > 0 && sorted[5] > 0 && sorted[10] > 0 {
    //         for index_2 in 0..10 {
    //             print!("{} ", sorted[index_2]);
    //         }
    //         println!("");
    //     }
        
    //     //println!("{:?}", sorted[0..10]);
    //     all_sorted.push(sorted);
    // }


    println!("End Data Loading");
    (all_cell_data, cell_ids)
}

fn data_loader_scrna() {
    println!("Start Data Loading");
    let data_file = File::open(DATA_FILE_2).expect("cannot open data file");
    let mut data_reader = BufReader::new(&data_file);
    let mut all_cell_data= vec![];
    let mut gene_expressed_by_cells = vec![];
    for (line_index, line) in data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        //all_cell_data.push(CellData::new(values.len()));
        if line_index == 0 {
            gene_expressed_by_cells = vec![0; values.len()];
        }
        for (gene_index, value) in values.iter().enumerate() {
            let read_count = (value.to_string().parse::<f32>().unwrap() * 10.0) as u16;
            if read_count > 1 {
                gene_expressed_by_cells[gene_index] += 1;
            }
        }
    }
    gene_expressed_by_cells.sort();
    println!("{}", gene_expressed_by_cells.len());
    println!("{}", gene_expressed_by_cells[gene_expressed_by_cells.len() / 2]);
    println!("{}", gene_expressed_by_cells[gene_expressed_by_cells.len() - 1]);
    let mut count = 0;
    for gene in gene_expressed_by_cells {
        if gene > 30000 {  
            count += 1;
        }
    }
    println!("count {}", count);
    
    let mut data_reader = BufReader::new(&data_file);
    for (line_index, line) in data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        all_cell_data.push(CellData::new(values.len()));
        for (gene_index, value) in values.iter().enumerate() {
            // convert to u32 and add to cell data
            //let read_count = (value.to_string().parse::<f32>().unwrap() * 10.0) as u16;
            //all_cell_data[line_index].read_counts.push(read_count);
        }
    }
    // sort the all data based on one gene entry just to see
    // let mut all_sorted = vec![];
    // let mut indices: Vec<usize> = (0..all_cell_data[0].read_counts.len()).collect();
    
    // indices.sort_by_key(|&i| all_cell_data[0].read_counts[i]);
    // indices.reverse();
    // for index in 0..all_cell_data.len() {
    //     let sorted: Vec<_> = indices.iter().map(|&i| all_cell_data[index].read_counts[i]).collect();
    //     if sorted[0] > 0 && sorted[5] > 0 && sorted[10] > 0 {
    //         for index_2 in 0..10 {
    //             print!("{} ", sorted[index_2]);
    //         }
    //         println!("");
    //     }
        
    //     //println!("{:?}", sorted[0..10]);
    //     all_sorted.push(sorted);
    // }


    println!("End Data Loading");
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
    //println!("{:?}", gene_non_zero_cluster);
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


fn _result_display(log_loss_final: &Vec<Vec<f32>>, ground_truth: &Vec<usize>) {
    let mut method_assignment = vec![];
    for (_index, final_log_probability) in log_loss_final.iter().enumerate() {
        let index_of_max: usize = final_log_probability.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
        //println!("cell {} assigned {}", index, index_of_max);
        method_assignment.push(index_of_max);
    }
    //sleep(Duration::from_secs(2));
    let rand_index = _rand_index_calculator(&method_assignment, ground_truth);
    println!("Rand Index {}", rand_index);
}

fn _rand_index_calculator (predicted: &Vec<usize>, ground_truth: &Vec<usize>) -> f64 {
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
    gene_count: usize
}
impl CellData {
    fn new(gene_count: usize) -> CellData {
        CellData{
            read_counts: vec![0; gene_count],
            gene_count: gene_count,
        }
    }
}