use numpy::PyReadonlyArray1;
use pyo3::{FromPyObject, pymodule};

#[derive(FromPyObject)]
pub struct IntegrationParams<'a> {
    pub k: PyReadonlyArray1<'a, f64>,
    pub curvature_spectrum: PyReadonlyArray1<'a, f64>,
    pub u_max: Option<f64>,
    pub du: Option<f64>,
    pub dv: Option<f64>,
}

impl<'a> IntegrationParams<'a> {
    pub fn du(&self) -> f64 {
        self.du.unwrap_or(1e-3)
    }
    pub fn u_max(&self) -> f64 {
        self.u_max.unwrap_or(1e2)
    }
    pub fn dv(&self) -> f64 {
        self.dv.unwrap_or(1e-3)
    }
}

#[pymodule]
mod pyinflat {
    use pyo3::{Py, Python, pyfunction};
    use numpy::{PyArray1, PyReadonlyArray1};
    use inflat::igw::sigw_2_spectrum;

    #[pyfunction]
    #[pyo3(signature = (k, curvature_spectrum, u_max=None, du=None, dv=None))]
    pub fn sigw_energy_density(
        py: Python,
        k: PyReadonlyArray1<'_, f64>,
        curvature_spectrum: PyReadonlyArray1<'_, f64>,
        u_max: Option<f64>,
        du: Option<f64>,
        dv: Option<f64>,
    ) -> Py<PyArray1<f64>> {
        let k_data = k.as_slice().unwrap();
        let spectrum = curvature_spectrum.as_slice().unwrap();
        let result = sigw_2_spectrum(k_data, spectrum, u_max.unwrap_or(1e2), du.unwrap_or(0.05), dv.unwrap_or(0.05), |_, _| {});
        PyArray1::from_vec(py, result).to_owned().unbind()
    }
}
