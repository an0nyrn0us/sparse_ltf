use lycaon::booster::Booster;
use lycaon::booster::LPBoost;
// use lycaon::base_learner::DStump;
use lycaon::read_csv;

use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;


pub mod stump;
pub mod ltf;
pub mod average;

use ltf::{BddLtf, NaiveLtf};


fn main() {
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("zdd_summary.csv").unwrap();
        let line = "dataset,\
                    sample_size,\
                    feature_size,\
                    name,\
                    running_time,\
                    node,\
                    edge,\
                    path\n";
        file.write_all(line.as_bytes()).unwrap();
    }
    // Path to datasets
    let path = "/home/mitsuboshi/dataset/benchmarks/shrinked/";

    let datasets = [
        "banana", "breast_cancer", "diabetis", "flare_solar", "german",
        "heart", "image", "ringnorm", "splice", "thyroid",
        "titanic", "twonorm", "waveform",
    ];

    for dataset in datasets {
        // Concat the path and dataset
        println!("{dataset}");
        let name = format!("{}{}_train.csv", path, dataset);


        // Read the LIBSVM file
        let sample = read_csv(name).unwrap();


        let m   = sample.len() as u64;
        let dim = sample.feature_len() as u64;

        // Initialize the base learner
        let start   = Instant::now();
        let bdd_ltf = BddLtf::init(&sample);
        let finish  = Instant::now();

        let running_time = (finish - start).as_secs_f64();
        let node = bdd_ltf.nodes.len() as u64 + 1;
        let edge = node * 2 - 2;
        let path = bdd_ltf.count_ltfs();
        println!("\tRunning time: {running_time} sec");
        println!(
            "\t# of nodes: {node},\n\t# of edges: {edge}, path: {path}"
        );


        write2file(
            &dataset, m, dim, "ZddLtf", running_time, node, edge, path
        ).unwrap();
    }
}


fn write2file(dataset: &str,
              m:        u64,
              dim:      u64,
              name:    &str,
              runtime:  f64,
              node:     u64,
              edge:     u64,
              path:     u64)
    -> std::io::Result<()>
{
    // The output file name
    let output = "zdd_summary.csv";

    let mut file = OpenOptions::new()
        .append(true)
        .open(output)?;


    let line = format!(
        "{dataset},{m},{dim},{name},{runtime},{node},{edge},{path}\n"
    );


    file.write_all(line.as_bytes())?;


    Ok(())
}

