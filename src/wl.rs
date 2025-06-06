use std::io::{self, Write};

use num_complex::{Complex, Complex64};
use num_traits::{Float, FromPrimitive};

pub trait WlEncode {
    fn encode<W: Write>(&self, write: &mut W) -> io::Result<()>;
}

impl WlEncode for f64 {
    fn encode<W: Write>(&self, write: &mut W) -> io::Result<()> {
        let (mantissa, exp, sign) = Float::integer_decode(*self);
        write.write_fmt(format_args!("N[{} * {} * 2^{}]", sign, mantissa, exp))
    }
}

impl<T: WlEncode> WlEncode for Complex<T>
where
    T: PartialOrd + FromPrimitive,
{
    fn encode<W: Write>(&self, write: &mut W) -> io::Result<()> {
        self.re.encode(write)?;
        write.write_all(b"+")?;
        self.im.encode(write)?;
        write.write_all(b"*I")?;
        Ok(())
    }
}

impl WlEncode for [f64] {
    fn encode<W: Write>(&self, write: &mut W) -> io::Result<()> {
        write.write_all(b"Developer`ToPackedArray@{")?;
        for i in 0..self.len() {
            if i > 0 {
                write.write(b",")?;
            }
            self[i].encode(write)?;
        }
        write.write_all(b"}")?;
        Ok(())
    }
}

impl WlEncode for [Complex64] {
    fn encode<W: Write>(&self, write: &mut W) -> io::Result<()> {
        write.write_all(b"Developer`ToPackedArray@{")?;
        for i in 0..self.len() {
            if i > 0 {
                write.write(b",")?;
            }
            self[i].encode(write)?;
        }
        write.write_all(b"}")?;
        Ok(())
    }
}

pub enum FunctionHead<'a> {
    List,
    Association,
    Other(&'a str),
}

pub fn write_array<T, W: Write>(arr: &[T], write: &mut W, packed: bool) {}
