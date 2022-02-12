use lycaon::booster::Booster;
use lycaon::booster::LPBoost;
use lycaon::base_learner::DStump;
use lycaon::read_csv;
use lycaon::data_type::Sample;

use std::time::Instant;
use std::fs::OpenOptions;
use std::io::Write;


pub mod stump;
pub mod average;
// pub mod zdd_softboosting;

// use zdd_softboosting::*;

use stump::ZDDStump;


fn main() {
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open("stump_test_error.csv").unwrap();
        let line = "dataset,\
                    sample_size,\
                    feature_size,\
                    stump,\
                    capping_ratio,\
                    pre_process_time,\
                    running_time,\
                    test_error,\
                    optimal_value\n";
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
        println!("*Dataset: {dataset}");
        // Concat the path and dataset
        let name = format!("{path}{dataset}_train.csv");


        // Read the LIBSVM file
        let sample = read_csv(name).unwrap();

        let m   = sample.len() as u64;
        let dim = sample.feature_len() as u64;


        let name = format!("{path}{dataset}_test.csv");
        let test_sample = read_csv(name).unwrap();


        // Initialize parameter.
        // We fix the tolerance parameter as 0.01.
        let tolerance = 0.01_f64;

        let capping_ratios = (1..9).map(|n| n as f64 * 0.1)
            .collect::<Vec<f64>>();


        {
            // Initialize the base learner
            let start  = Instant::now();
            let zddstump = ZDDStump::init(&sample);
            let finish = Instant::now();

            let pre_process_time = (finish - start).as_secs_f64();
            println!("\t\tPre process: {} sec", pre_process_time);
            for capping_ratio in capping_ratios.iter() {
                // Initialize the capping parameter.
                let cap = sample.len() as f64 * *capping_ratio;


                println!(
                    "\tRunning ZDDStump with capping: {capping_ratio}"
                );


                println!("\tRunning LPBoost for ZDDStump...");
                let mut lpboost = LPBoost::init(&sample).capping(cap);


                let start  = Instant::now();
                lpboost.run(&zddstump, &sample, tolerance);
                let finish = Instant::now();

                let running_time = (finish - start).as_secs_f64();

                println!("\t\tOptimal value: {}", lpboost.gamma_hat);
                println!("\t\tRunning time:  {} sec", running_time);


                let loss = test_error(&test_sample, &lpboost);


                write2file(
                    &dataset,
                    m,
                    dim,
                    "ZDDStump",
                    *capping_ratio,
                    pre_process_time,
                    running_time,
                    loss,
                    lpboost.gamma_hat
                ).unwrap();
            }
        }




            // LPBoost with Decision Stump compressed to ZDD.
        {
            // Initialize the base learner
            let start  = Instant::now();
            let dstump = DStump::init(&sample);
            let finish = Instant::now();

            let pre_process_time = (finish - start).as_secs_f64();
            println!("\t\tPre process: {pre_process_time} sec");

            for capping_ratio in capping_ratios {
                println!(
                    "\tRunning DStump with capping: {capping_ratio}"
                );
                // Initialize the capping parameter.
                let cap = sample.len() as f64 * capping_ratio;

                println!("\tRunning LPBoost for DStump...");
                let mut lpboost = LPBoost::init(&sample).capping(cap);



                let start  = Instant::now();
                lpboost.run(&dstump, &sample, tolerance);
                let finish = Instant::now();

                let running_time = (finish - start).as_secs_f64();

                println!("\t\tOptimal value: {}", lpboost.gamma_hat);
                println!("\t\tRunning time:  {} sec", running_time);

                let loss = test_error(&test_sample, &lpboost);


                write2file(
                    &dataset,
                    m,
                    dim,
                    &"DStump",
                    capping_ratio,
                    pre_process_time,
                    running_time,
                    loss,
                    lpboost.gamma_hat
                ).unwrap();
            }
        }
    }
}


fn write2file(dataset:    &str,
              m:           u64,
              dim:         u64,
              stump:      &str,
              capping:     f64,
              pre_process: f64,
              running:     f64,
              test_err:    f64,
              opt_val:     f64)
    -> std::io::Result<()>
{
    // The output file name
    let output = "stump_test_error.csv";

    let mut file = OpenOptions::new()
        .append(true)
        .open(output)?;


    let line = format!(
        "{dataset},{m},{dim},{stump},{capping},\
         {pre_process},{running},{test_err},{opt_val}\n",
    );


    file.write_all(line.as_bytes())?;


    Ok(())
}


fn test_error<B: Booster<f64, f64>>(sample: &Sample<f64, f64>, booster: &B) -> f64 {
    let cumulative_loss = sample.iter()
        .fold(0.0, |mut acc, ex| {
            let prediction = booster.predict(&ex.data);

            if prediction != ex.label { acc += 1.0; }
            acc
        });

    cumulative_loss / sample.len() as f64
}
