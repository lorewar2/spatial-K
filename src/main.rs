use std::fs::File;
use std::io::{BufRead, BufReader};
use rand::Rng;

const DATA_FILE: &'static str = "./data/data.csv";
const K: usize = 8;

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
    let num_clusters = K;
    let log_prior: f32 = (1.0 / (num_clusters as f32)).ln();
    let mut last_log_loss = 0.0;
    // update probs to update the final probabilities
    let mut update_prob: Vec<Vec<f32>> = Vec::new();
    for cluster in 0..num_clusters {
        update_prob.push(Vec::new());
        for _index in 0..gene_count {
            update_prob[cluster].push(0.5);
        }
    }
    // run 10 times and see
    for run in 0..10 {
        let mut log_poisson_total = 0.0;
        // reset
        reset_update_prob(num_clusters, gene_count, &mut update_prob);
        for (_celldex, cell) in all_cell_data.iter().enumerate() {
            // calculate poisson loss here
            let log_poisson = poisson_loss(cell, &cluster_centers, log_prior);
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
    }
}

fn update_cluster_centers(gene_count: usize, update_prob: &Vec<Vec<f32>>, cluster_centers: &mut Vec<Vec<f32>>) {
    for locus in 0..gene_count {
        for cluster in 0..update_prob.len() {
            let update = update_prob[cluster][locus];
            cluster_centers[cluster][locus] = update.min(0.99).max(0.01);
        }
    }
}

fn update_update_prob(update_prob: &mut Vec<Vec<f32>>, cell: &CellData, log_poisson:  &Vec<f32>) {
    for locus in 0..cell.gene_count {
        for (cluster, probability) in log_poisson.iter().enumerate() {
            update_prob[cluster][locus] += probability * (cell.read_counts[locus] as f32);
        }
    }
}

fn poisson_loss(cell: &CellData, cluster_centers: &Vec<Vec<f32>>, log_prior: f32) -> Vec<f32> {
    let mut log_probabilities: Vec<f32> = Vec::new();
    for (cluster, center) in cluster_centers.iter().enumerate() {
        log_probabilities.push(log_prior);
        for (gene_index, read_count) in cell.read_counts.iter().enumerate() {
            //L=λ​−x​log(λ​)+log(x​!)
            //x = read count 
            //λ = center[gene_index]
            log_probabilities[cluster] += center[gene_index] - *read_count as f32 * center[gene_index].ln();
        }
    }
    log_probabilities
}

fn log_sum_exp(p: &Vec<f32>) -> f32{
    let max_p: f32 = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_rst: f32 = p.iter().map(|x| (x - max_p).exp()).sum();
    max_p + sum_rst.ln()
}

fn reset_update_prob(num_clusters: usize, gene_count: usize, update_prob: &mut Vec<Vec<f32>>) {
    for cluster in 0..num_clusters {
        for index in 0..gene_count {
            update_prob[cluster][index] = 0.5;
        }
    }
}

fn init_cluster_centers_uniform(gene_count: usize, num_clusters: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
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