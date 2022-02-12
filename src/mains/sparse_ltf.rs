use lycaon::booster::Booster;
use lycaon::booster::LPBoost;
// use lycaon::base_learner::DStump;
use lycaon::read_csv;

use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;


pub mod sparse_ltf;

use sparse_ltf::*;



fn main() {
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("sparseltf.csv").unwrap();
        let line = "dataset,\
                    sample_size,\
                    feature_size,\
                    base_learner,\
                    capping_ratio,\
                    pre_process_time,\
                    running_time,\
                    test_error,\
                    optimal_value\n";
        file.write_all(line.as_bytes()).unwrap();



        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("sparseltf_summary.csv").unwrap();
        let line = "dataset,\
                    sample_size,\
                    feature_size,\
                    running_time,\
                    node,\
                    edge,\
                    path\n";
        file.write_all(line.as_bytes()).unwrap();
    }
    // Path to datasets
    let path = "/home/mitsuboshi/dataset/benchmarks/splitted/";

    let datasets = [
        "banana", "breast_cancer", "diabetis", "flare_solar", "german",
        "heart", "image", "ringnorm", "splice", "thyroid",
        "titanic", "twonorm", "waveform",
    ];

    for dataset in datasets {
        // Concat the path and dataset
        let name = format!("{}{}_train.csv", path, dataset);


        // Read the LIBSVM file
        let sample = read_csv(name).unwrap();

        let m   = sample.len() as u64;
        let dim = sample.feature_len() as u64;

        let name = format!("{}{}_test.csv", path, dataset);
        let test_sample = read_csv(name).unwrap();

        // Since the base learner does not depend on the capping ratio,
        // it is enough to initialize once.
        // Initialize the base learner
        let start  = Instant::now();
        let sparse = SparseLtf::init(&sample);
        let finish = Instant::now();

        let pre_process_time = (finish - start).as_secs_f64();
        println!("Pre process: {} sec", pre_process_time);
        {
            let node_size = sparse.nodes.len() as u64 + 1;
            let edge_size = sparse.nodes.iter()
                .fold(0_u64, |mut acc, node| {
                    if node.borrow().neg.is_some() {
                        acc += 1;
                    }
                    if node.borrow().pos.is_some() {
                        acc += 1;
                    }
                    acc
                });
            let path = sparse.count_ltfs();

            dd_summary(
                &dataset, m, dim, "sparseltf", pre_process_time,
                node_size, edge_size, path
            ).unwrap();
        }


        // Initialize parameter.
        // We fix the tolerance parameter as 0.01.
        let tolerance = 0.01_f64;

        let capping_ratios = (1..9).map(|n| n as f64 * 0.1)
            .collect::<Vec<f64>>();


        for capping_ratio in capping_ratios {
            // Initialize the capping parameter.
            let cap = sample.len() as f64 * capping_ratio;


            println!(
                "Dataset: {dataset}, m is: {m},\
                 dim is: {dim}, cap: {capping_ratio}"
            );


            // LPBoost with Decision Stump compressed to ZDD.
            println!("\tRunning LPBoost for SparseLtf...");
            let mut lpboost = LPBoost::init(&sample).capping(cap);



            let start  = Instant::now();
            lpboost.run(&sparse, &sample, tolerance);
            let finish = Instant::now();

            let running_time = (finish - start).as_secs_f64();


            let loss = test_sample.iter()
                .fold(0.0, |acc, example| {
                    let prediction = lpboost.predict(&example.data);

                    if prediction != example.label {
                        acc + 1.0
                    } else {
                        acc
                    }
                });
            let loss = loss / test_sample.len() as f64;

            println!("\t\tOptimal value:   {}", lpboost.gamma_hat);
            println!("\t\tTest error(0/1): {}", loss);
            println!("\t\tRunning time:    {} sec", running_time);


            write2file(
                &dataset,
                m,
                dim,
                "SparseLtf",
                capping_ratio,
                pre_process_time,
                running_time,
                loss,
                lpboost.gamma_hat
            ).unwrap();
        }
    }
}


fn write2file(dataset:     &str,
              m:            u64,
              dim:          u64,
              baselearner: &str,
              capping:      f64,
              pre_process:  f64,
              running:      f64,
              loss:         f64,
              opt_val:      f64)
    -> std::io::Result<()>
{
    // The output file name
    let output = "sparseltf.csv";

    let mut file = OpenOptions::new()
        .append(true)
        .open(output)?;


    let line = format!(
        "{dataset},{m},{dim},{baselearner},{capping},\
         {pre_process},{running},{loss},{opt_val}\n",
    );


    file.write_all(line.as_bytes())?;


    Ok(())
}


fn dd_summary(dataset:      &str,
              m:             u64,
              dim:           u64,
              baselearner:  &str,
              running_time:  f64,
              node:          u64,
              edge:          u64,
              path:          u64)
    -> std::io::Result<()>
{
    // The output file name
    let output = "sparseltf_summary.csv";

    let mut file = OpenOptions::new()
        .append(true)
        .open(output)?;


    let line = format!(
        "{dataset},{m},{dim},{baselearner},{running_time},\
         {node},{edge},{path}\n",
    );


    file.write_all(line.as_bytes())?;


    Ok(())
}


#[test]
fn shrinked_test() {

    let path = "/Users/ryotaromitsuboshi/Documents/Datasets/german.csv";

    let sample = read_csv(path).unwrap();

    let start  = Instant::now();
    let sparse = SparseLtf::init(&sample);
    let finish = Instant::now();

    let pre_process_time = (finish - start).as_secs_f64();
    println!("\t\tPre process: {} sec", pre_process_time);
    // LPBoost with Decision Stump compressed to ZDD.
    println!("\tRunning LPBoost for SparseLtf...");
    let mut lpboost = LPBoost::init(&sample)
        .capping(0.8 * sample.len() as f64);
    let start  = Instant::now();
    lpboost.run(&sparse, &sample, 0.01);
    let finish = Instant::now();

    let running_time = (finish - start).as_secs_f64();

    println!("\t\tOptimal value(LIBSVM): {}", lpboost.gamma_hat);
    println!("\t\tRunning time: {} sec", running_time);

    assert!(true);
    // println!("# of hypothesis: {}", sparse.count_ltfs());
}
