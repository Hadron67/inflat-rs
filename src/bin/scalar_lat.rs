use std::{fs::create_dir_all, time::Duration};

use anyhow::Ok;
use bincode::{Decode, Encode};
use inflat::{
    background::BINCODE_CONFIG,
    c2fn::C2Fn,
    lat::{Lattice, LatticeParam},
    models::QuadraticPotential,
    scalar::{ScalarFieldParams, ScalarFieldState},
    util::{RateLimiter, VecN, lazy_file, limit_length},
};
use plotly::{Layout, Plot, Scatter, common::ExponentFormat, layout::Axis};

struct Params<F> {
    pub scalar_params: ScalarFieldParams<3, F>,
    pub phi: f64,
    pub v_phi: f64,
    pub dt: f64,
    pub end_n: f64,
}

#[derive(Encode, Decode, Debug, Clone, Copy)]
struct Measurables {
    pub a: f64,
    pub phi: f64,
    pub v_phi: f64,
}

#[derive(Encode, Decode)]
struct IntermediateData {
    pub evaluation_measurables: Vec<Measurables>,
    pub final_state: ScalarFieldState<3>,
}

impl<F> Params<F>
where
    F: C2Fn<f64, Output = f64> + Send + Sync,
{
    pub fn run(&self, out_dir: &str) -> anyhow::Result<()> {
        create_dir_all(out_dir)?;
        let int_data = lazy_file(
            &format!("{}/int_data.bincode", out_dir),
            BINCODE_CONFIG,
            || {
                let mut field = ScalarFieldState::zeros(self.scalar_params.lattice.size);
                self.scalar_params
                    .init(&mut field, 1.0, self.phi, self.v_phi);
                let mut rate_limiter = RateLimiter::new(Duration::from_millis(100));
                let mut measurables = vec![Measurables {
                    a: 1.0,
                    phi: self.phi,
                    v_phi: self.v_phi,
                }];
                while self.scalar_params.scale_factor(&field).ln() < self.end_n {
                    self.scalar_params.apply_full_k_order2(&mut field, self.dt);
                    let m = Measurables {
                        a: self.scalar_params.scale_factor(&field),
                        phi: field.phi.average(),
                        v_phi: self.scalar_params.v_phi(&field),
                    };
                    rate_limiter.run(|| println!("{:?}", &m));
                    measurables.push(m);
                }
                IntermediateData {
                    evaluation_measurables: measurables,
                    final_state: field,
                }
            },
        )?;
        let max_length = 500000;
        {
            let mut plot = Plot::new();
            let mut efoldings = vec![];
            let mut phi = vec![];
            let mut v_phi = vec![];
            for state in limit_length(int_data.evaluation_measurables, max_length) {
                efoldings.push(state.a.ln());
                phi.push(state.phi);
                v_phi.push(state.v_phi);
            }
            plot.add_trace(Scatter::new(efoldings.clone(), phi).name("phi"));
            plot.add_trace(
                Scatter::new(efoldings.clone(), v_phi)
                    .name("v_phi")
                    .y_axis("y2"),
            );
            plot.set_layout(
                Layout::new()
                    .x_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis(Axis::new().exponent_format(ExponentFormat::Power))
                    .y_axis2(Axis::new().exponent_format(ExponentFormat::Power)),
            );
            plot.write_html(&format!("{}/background.html", out_dir));
        }
        Ok(())
    }
}

pub fn main() {
    let params = Params {
        scalar_params: ScalarFieldParams {
            kappa: 1.0,
            potential: QuadraticPotential::new(0.51e-5),
            lattice: LatticeParam {
                spacing: VecN::new([0.1; 3]),
                size: VecN::new([16; 3]),
            },
        },
        phi: 14.5,
        v_phi: -0.8152 * 0.51e-5,
        dt: 1.0,
        end_n: 7.0,
    };
    params.run("out/scalar_lat.set1/").unwrap();
}
