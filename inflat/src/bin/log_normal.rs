use std::f64::consts::PI;

use inflat::{igw::sigw_2_spectrum, util::ParamRange};
use plotly::{Plot, Scatter};

pub fn make_log_normal_spec(k_range: ParamRange<f64>, k_star: f64, sigma: f64, amp: f64) -> (Vec<f64>, Vec<f64>) {
    let k_data = k_range.as_logspace().collect();
    let spec = k_range.as_logspace().map(|k|amp / (2.0 * PI).sqrt() * (-(k / k_star).ln().powi(2) / 2.0 / sigma / sigma).exp()).collect();
    (k_data, spec)
}

pub fn main() {
    let (k_data, spectrum) = make_log_normal_spec(ParamRange::new(1.0, 1000.0, 100), 1000.0_f64.sqrt(), 10.0, 100.0);
    let spec = sigw_2_spectrum(&k_data, &spectrum, 100.0, 0.05, 0.05, |_, _|{});
    let mut plot = Plot::new();
    plot.add_trace(Scatter::new(k_data, spec));
}
