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
pub mod sparse_ltf;

use ltf::{BddLtf, NaiveLtf};


fn main() {
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("naive_ltf.csv").unwrap();
        let line = "dataset,\
                    sample_size,\
                    feature_size,\
                    stump,\
                    capping_ratio,\
                    pre_process_time,\
                    running_time,\
                    iterations,\
                    test_error,\
                    optimal_value\n";
        file.write_all(line.as_bytes()).unwrap();
    }
    // Path to datasets
    let path = "/home/mitsuboshi/dataset/benchmarks/shrinked/";

    let datasets = [
        "banana", "breast_cancer", "diabetis", "flare_solar", "german",
        "heart", "image", "ringnorm", "splice", "thyroid",
        "twonorm", "waveform",
    ];
    for dataset in datasets {
        // Concat the path and dataset
        let name = format!("{}{}_train.csv", path, dataset);


        // Read the LIBSVM file
        let sample = read_csv(name).unwrap();

        let name = format!("{}{}_test.csv", path, dataset);
        let test_sample = read_csv(name).unwrap();

        let m   = sample.len() as u64;
        let dim = sample.feature_len() as u64;



        // Initialize parameter.
        // We fix the tolerance parameter as 0.01.
        let tolerance = 0.01_f64;

        let capping_ratios = (1..9).map(|n| n as f64 * 0.1)
            .collect::<Vec<f64>>();


        for capping_ratio in capping_ratios {
            // Initialize the capping parameter.
            let cap = sample.len() as f64 * capping_ratio;


            println!(
                "Dataset: {}, sample size: {}, sample features: {}, cap: {}",
                dataset, m, dim, capping_ratio
            );


            // LPBoost with Decision Stump compressed to ZDD.
            println!("\tRunning LPBoost for DStump...");
            let mut lpboost = LPBoost::init(&sample).capping(cap);

            // Initialize the base learner
            let start  = Instant::now();
            let base_learner = NaiveLtf::init(&sample);
            let finish = Instant::now();

            let pre_process_time = (finish - start).as_secs_f64();
            println!("\t\tPre process: {} sec", pre_process_time);


            let start  = Instant::now();
            lpboost.run(&base_learner, &sample, tolerance);
            let finish = Instant::now();

            let running_time = (finish - start).as_secs_f64();
            let iterations = lpboost.classifiers.len() as u64;


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
                &"NaiveLtf",
                capping_ratio,
                pre_process_time,
                running_time,
                iterations,
                loss,
                lpboost.gamma_hat
            ).unwrap();
        }
    }

}


fn write2file(dataset: &str,
              m:        u64,
              dim:      u64,
              stump:   &str,
              capping:  f64,
              pre_process: f64,
              running:  f64,
              iterate:  u64,
              loss:     f64,
              opt_val:  f64)
    -> std::io::Result<()>
{
    // The output file name
    let output = "naive_ltf.csv";

    let mut file = OpenOptions::new()
        .append(true)
        .open(output)?;


    let line = format!(
        "{dataset},{m},{dim},{stump},{capping},\
        {pre_process},{running},{iterate},{loss},{opt_val}\n",
    );


    file.write_all(line.as_bytes())?;


    Ok(())
}
