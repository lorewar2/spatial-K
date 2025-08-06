use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

const DATA_FILE: &'static str = "./data/data.csv";
const K: usize = 8;
const SEED: u64 = 10;

fn main() {
    // get the data in form of cell and locus
    let all_cell_data = data_loader();
    // check
    assert!(all_cell_data.last().unwrap().read_counts.len() == all_cell_data.first().unwrap().gene_count);
    let gene_count = all_cell_data[0].gene_count;
    // initialize the cluster centers randomly, for now
    let mut cluster_centers = init_cluster_centers_uniform(gene_count, K);
    // do em with poisson
    em(gene_count, &mut cluster_centers, &all_cell_data);
}

fn em(gene_count: usize, mut cluster_centers: &mut Vec<Vec<f32>>, all_cell_data: &Vec<CellData>) {
    // vec to save the log loss for each cluster from each cell, to determine the one with least
    let mut log_loss_final = vec![vec![]; all_cell_data.len()];
    let num_clusters = K;
    // log prior assume all clusters are equal
    let log_prior: f32 = (1.0 / (num_clusters as f32)).ln();
    let mut last_log_loss = 0.0;
    // update probs to update the cc
    let mut update_prob: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..num_clusters {
        update_prob.push(Vec::new());
        for _index in 0..gene_count {
            update_prob[cluster].push(0.0001);
        }
    }
    // run 10 times and see
    for run in 0..10 {
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
            update_update_prob(&mut update_prob, cell, &log_poisson);
        }
        update_cluster_centers(gene_count, &update_prob, &mut cluster_centers);
        // display stuff
        let log_loss_change = log_poisson_total - last_log_loss;
        last_log_loss = log_poisson_total;
        println!("poisson\t{}\t{}\t{}", run, log_poisson_total, log_loss_change);
        // check the cluster assignment
        let mut assigned_vec: Vec<usize> = vec![0; num_clusters];
        for final_log_probability in &log_loss_final {
            let index_of_max: usize = final_log_probability.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).map(|(index, _)| index).unwrap();
            assigned_vec[index_of_max] += 1;
        }
        println!("Assignment vec {:?}", assigned_vec);
    }
}

fn update_cluster_centers(gene_count: usize, update_prob: &Vec<Vec<f32>>, cluster_centers: &mut Vec<Vec<f32>>) {
    for locus in 0..gene_count {
        for cluster in 0..update_prob.len() {
            let update = update_prob[cluster][locus] / 1000.0;
            //println!("{}", update);
            cluster_centers[cluster][locus] = update.min(0.99).max(0.01);
        }
    }
}

fn update_update_prob(update_prob: &mut Vec<Vec<f32>>, cell: &CellData, log_poisson:  &Vec<f32>) {
    for locus in 0..cell.gene_count {
        // get the sum
        let sum = log_sum_exp(log_poisson);
        for (cluster, probability) in log_poisson.iter().enumerate() {
            // normalize and turn log to normal
            let update_prob_exp = (probability - sum).exp();
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
            //println!("fact {} others {}", (factorial(*read_count) as f32).ln(), - center[gene_index] + *read_count as f32 * center[gene_index].ln());
            let value = - center[gene_index] + *read_count as f32 * center[gene_index].ln();
            let _factorial = (factorial(*read_count) as f32).ln();
            log_probabilities[cluster] += value;// - _factorial;
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

fn init_cluster_centers_uniform(gene_count: usize, num_clusters: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    // Generate random numbers using the seeded RNG
    let mut centers: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..num_clusters {
        centers.push(Vec::new());
        for _ in 0..gene_count {
            centers[cluster].push(rng.gen::<f32>().min(0.9999).max(0.0001));
        }
    }
    centers
}

fn data_loader() -> Vec<CellData> {
    println!("Start Data Loading");
    let data_file = File::open(DATA_FILE).expect("cannot open data file");
    let data_reader = BufReader::new(data_file);
    let mut all_cell_data= vec![];
    for (line_index, line) in data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        if line_index == 0 {
            // this is the header, make new cell vector
            all_cell_data = vec![CellData::new(); values.len()];
            continue;
        }
        for (cell_index, value) in values.iter().enumerate() {
            // convert to u32 and add to cell data
            let read_count = value.to_string().parse::<u16>().unwrap();
            all_cell_data[cell_index].read_counts.push(read_count);
            all_cell_data[cell_index].gene_count = line_index;
        }
    }
    println!("End Data Loading");
    all_cell_data
}

#[derive(Clone)]
struct CellData {
    read_counts: Vec<u16>,
    gene_count: usize
}
impl CellData {
    fn new() -> CellData {
        CellData{
            read_counts: Vec::new(),
            gene_count: 0,
        }
    }
}