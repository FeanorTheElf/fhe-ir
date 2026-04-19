use std::cell::OnceCell;
use std::fmt::{Debug, Display};
use std::str::FromStr;

use elsa::sync::FrozenMap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Identifier<'a> {
    value: &'a str
}

impl<'a> Identifier<'a> {

    pub fn parse_leading_identifier(data: &str, table: &'a IdentifierTable) -> Option<(Self, usize)> {
        if !data.starts_with("%") {
            return None;
        }
        let idx = data.chars().skip(1).take_while(|c| c.is_alphanumeric() || *c == '_').count();
        if idx > 0 {
            Some((table.internalize(&data[..idx]), idx))
        } else {
            None
        }
    }
}


impl<'a> Display for Identifier<'a> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.value)
    }
}

impl<'a> Debug for Identifier<'a> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

pub struct IdentifierTable {
    data: FrozenMap<String, String>
}

impl IdentifierTable {

    pub fn new() -> Self {
        IdentifierTable { data: FrozenMap::new() }
    }

    pub fn internalize<'a>(&'a self, val: &str) -> Identifier<'a> {
        if let Some(res) = self.data.get(val) {
            Identifier { value: res }
        } else {
            Identifier { value: self.data.insert(val.to_owned(), val.to_owned()) }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Instruction<'a, Ptx = Identifier<'a>> {
    AddCtxCtx { out: Identifier<'a>, in1: Identifier<'a>, in2: Identifier<'a> },
    AddPtxCtx { out: Identifier<'a>, in1: Identifier<'a>, in2: Ptx },
    MulCtxCtx { out: Identifier<'a>, in1: Identifier<'a>, in2: Identifier<'a> },
    MulPtxCtx { out: Identifier<'a>, in1: Identifier<'a>, in2: Ptx },
    MulIntCtx { out: Identifier<'a>, in1: Identifier<'a>, in2: i64 },
    Zero { out: Identifier<'a> },
    Return { val: Identifier<'a> },
    Galois { out: Vec<Identifier<'a>>, in1: Identifier<'a>, exponents: Vec<i64> },
    InnerProduct { out: Identifier<'a>, in1: Vec<Identifier<'a>>, in2: Vec<Ptx> }
}

impl<'a, Ptx: Display> Display for Instruction<'a, Ptx> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Instruction::*;
        match self {
            AddCtxCtx { out, in1, in2 } => write!(f, "{} = add {}, {}", out, in1, in2),
            AddPtxCtx { out, in1, in2 } => write!(f, "{} = add_ptx {}, {}", out, in1, in2),
            MulCtxCtx { out, in1, in2 } => write!(f, "{} = mul {}, {}", out, in1, in2),
            MulPtxCtx { out, in1, in2 } => write!(f, "{} = mul_ptx {}, {}", out, in1, in2),
            MulIntCtx { out, in1, in2 } => write!(f, "{} = mul_int {}, {}", out, in1, in2),
            Zero { out } => write!(f, "{} = zero", out),
            Galois { out, in1, exponents } => {
                let mut out_it = out.iter();
                if let Some(out) = out_it.next() {
                    write!(f, "{}", out)?;
                }
                for out in out_it {
                    write!(f, ", {}", out)?;
                }
                write!(f, " = galois {}, exponents = {:?}", in1, exponents)
            },
            Return { val } => write!(f, "return {}", val),
            InnerProduct { out, in1, in2 } => {
                write!(f, "{} = inner_prod ", out)?;
                for input in in1 {
                    write!(f, "{}, 0", input)?;
                }
                write!(f, "coefficients = [")?;
                let mut coeff_it = in2.iter();
                if let Some(coeff) = coeff_it.next() {
                    write!(f, "{}", coeff)?;
                }
                for coeff in coeff_it {
                    write!(f, ", {}", coeff)?;
                }
                write!(f, "]")
            }
        }
    }
}

impl<'a> Instruction<'a> {

    fn from_str(mut s: &str, table: &'a IdentifierTable) -> Option<Self> {
        let expect = |s: &mut &str, expected: &str| if s.starts_with(expected) {
            *s = &s[expected.len()..];
            return Some(());
        } else {
            return None;
        };
        let expect_ident = |s: &mut &str| if let Some((ident, continue_at)) = Identifier::parse_leading_identifier(*s, table) {
            *s = &s[continue_at..];
            return Some(ident);
        } else {
            return None;
        };
        let expect_int = |s: &mut &str| {
            let idx = s.chars().take_while(|c| *c == '-' || c.is_numeric()).count();
            if idx > 0 {
                let result = i64::from_str(*s).ok();
                *s = &s[idx..];
                return result;
            } else {
                return None;
            }
        };
        let expect_end = |s: &str, result| if s.len() == 0 {
            return Some(result);
        } else {
            return None;
        };

        s = s.trim();
        let mut outputs = Vec::new();
        while let Some(out) = expect_ident(&mut s) {
            outputs.push(out);
        }
        let expect_single_output = || if outputs.len() == 1 {
            return Some(outputs[0]);
        } else {
            return None;
        };

        if s.starts_with("add ") {
            let in1 = expect_ident(&mut s)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s)?;
            expect_end(s, Instruction::AddCtxCtx { out: expect_single_output()?, in1, in2 })
        } else if s.starts_with("add_ptx ") {
            let in1 = expect_ident(&mut s)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s)?;
            expect_end(s, Instruction::AddPtxCtx { out: expect_single_output()?, in1, in2 })
        } else if s.starts_with("mul ") {
            let in1 = expect_ident(&mut s)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s)?;
            expect_end(s, Instruction::MulCtxCtx { out: expect_single_output()?, in1, in2 })
        } else if s.starts_with("mul_ptx ") {
            let in1 = expect_ident(&mut s)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s)?;
            expect_end(s, Instruction::MulPtxCtx { out: expect_single_output()?, in1, in2 })
        } else if s.starts_with("mul_int ") {
            let in1 = expect_ident(&mut s)?;
            expect(&mut s, ", ")?;
            let in2 = expect_int(&mut s)?;
            expect_end(s, Instruction::MulIntCtx { out: expect_single_output()?, in1, in2 })
        } else if s.starts_with("galois ") {
            let in1 = expect_ident(&mut s)?;
            expect(&mut s, ", exponents = [")?;
            let mut exponents = Vec::new();
            while let Some(val) = expect_int(&mut s) {
                exponents.push(val);
            }
            expect(&mut s, "]")?;
            if exponents.len() != outputs.len() {
                return None;
            }
            expect_end(s, Instruction::Galois { out: outputs, in1, exponents })
        } else if s.starts_with("return ") {
            let val = expect_ident(&mut s)?;
            if outputs.len() == 0 {
                expect_end(s, Instruction::Return { val })
            } else {
                None
            }
        } else if s.starts_with("inner_prod ") {
            let mut values = Vec::new();
            while let Some(val) = expect_ident(&mut s) {
                values.push(val);
            }
            expect(&mut s, ", coefficients = [")?;
            let mut coefficients = Vec::new();
            while let Some(coeff) = expect_ident(&mut s) {
                coefficients.push(coeff);
            }
            expect(&mut s, "]")?;
            if coefficients.len() != values.len() {
                return None;
            }
            expect_end(s, Instruction::InnerProduct { out: expect_single_output()?, in1: values, in2: coefficients })
        } else if s.starts_with("zero") {
            expect_end(s, Instruction::Zero { out: expect_single_output()? })
        } else {
            None
        }
    }
}

struct ProgramData {
    ident_table: IdentifierTable,
    params: OnceCell<Vec<Identifier<'static>>>,
    instructions: OnceCell<Vec<Instruction<'static>>>
}

pub struct Program(Holder<'static, ProgramData>);

impl Display for Program {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "func (")?;
        let mut params_it = self.params.iter();
        if let Some(param) = params_it.next() {
            write!(f, "{}", param)?;
        }
        for param in params_it {
            write!(f, ", {}", param)?;
        }
        writeln!(f, ") {{")?;
        for inst in &self.instructions {
            writeln!(f, "    {}", inst)?;
        }
        writeln!(f, "}}")?;
        return Ok(());
    }
}

impl Program {

}