use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::io::{BufRead, BufReader, Read};
use std::ops::{Deref, DerefMut};
use std::str::FromStr;

use bimap::BiMap;

#[derive(Clone, PartialEq, Eq)]
pub struct PlaintextData {
    data: Vec<i64>
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum GenericInstruction<Ident = usize, Ptx = Ident> {
    AddCtxCtx { out: Ident, in1: Ident, in2: Ident },
    AddPtxCtx { out: Ident, in1: Ident, in2: Ptx },
    MulCtxCtx { out: Ident, in1: Ident, in2: Ident },
    MulPtxCtx { out: Ident, in1: Ident, in2: Ptx },
    MulIntCtx { out: Ident, in1: Ident, in2: i64 },
    Copy { out: Ident, in1: Ident },
    Zero { out: Ident },
    Return { val: Ident },
    Galois { out: Vec<Ident>, in1: Ident, exponents: Vec<i64> },
    InnerProduct { out: Ident, in1: Vec<Ident>, in2: Vec<Ptx> }
}

pub type Instruction<'a> = GenericInstruction<&'a str, &'a str>;
pub type InstructionWithData<'a, Ptx> = GenericInstruction<&'a str, &'a Ptx>;

#[derive(Clone, PartialEq, Eq)]
pub struct Program<Ptx = PlaintextData> {
    identifier_table: IdentifierTable,
    plaintext_table: HashMap<usize, Ptx>,
    inputs: Vec<usize>,
    instructions: Vec<GenericInstruction>
}

impl<Ptx> Program<Ptx> {

    pub fn new<'a, K, I>(inputs: &[&str], instructions: I, plaintext_table: HashMap<K, Ptx>) -> Self
        where I: IntoIterator<Item = Instruction<'a>>,
            K: Borrow<str>
    {
        let mut ident_table = IdentifierTable { counter: 0, mapping: BiMap::new() };
        let inputs = inputs.iter().map(|id| ident_table.ident_by_name(*id)).collect();
        let instructions = instructions.into_iter().map(|inst| inst.map_identifiers(&mut |id| ident_table.ident_by_name(id))).collect();
        let plaintext_table = plaintext_table.into_iter().map(|(k, v)| (ident_table.get_by_name(k.borrow()), v)).collect();
        return Self {
            identifier_table: ident_table,
            inputs: inputs,
            instructions: instructions,
            plaintext_table: plaintext_table
        };
    }

    pub fn inputs<'a>(&'a self) -> impl ExactSizeIterator<Item = &'a str> {
        self.inputs.iter().copied().map(|idx| self.identifier_table.get(idx))
    }

    pub fn instructions<'a>(&'a self) -> impl ExactSizeIterator<Item = Instruction<'a>> {
        self.instructions.iter().map(|inst| inst.clone().map_identifiers(&mut |idx| self.identifier_table.get(idx)))
    }

    pub fn instructions_with_data<'a>(&'a self) -> impl ExactSizeIterator<Item = InstructionWithData<'a, Ptx>> {
        self.instructions.iter().map(|inst| inst.clone().map_nonptx_identifiers(&mut |idx| self.identifier_table.get(idx)).map_ptx(&mut |key| self.plaintext_table.get(&key).unwrap()))
    }

    pub fn get_plaintext_data<'a>(&'a self, identifier: &str) -> Option<&'a Ptx> {
        self.identifier_table.get_by_name_opt(identifier).and_then(|idx| self.plaintext_table.get(&idx))
    }
}

impl Program {

    pub fn parse<R: Read>(data: R) -> Result<Self, usize> {
        Self::parse_impl(data, |mut s| {
            let mut data = Vec::new();
            expect(&mut s, "[")?;
            if let Some(val) = expect_int(&mut s) {
                data.push(val);
                while let Some(()) = expect(&mut s, ", ") {
                    data.push(expect_int(&mut s)?);
                }
            }
            expect(&mut s, "]")?;
            expect_end(s, PlaintextData::from(data))
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
struct IdentifierTable {
    counter: usize, 
    mapping: BiMap<usize, String> 
}

impl IdentifierTable {

    fn ident_by_name(&mut self, name: &str) -> usize {
        if let Some(res) = self.mapping.get_by_right(name) {
            return *res;
        } else {
            let idx = self.counter;
            self.counter += 1;
            self.mapping.insert(idx, name.to_owned());
            return idx;
        }
    }

    fn get(&self, idx: usize) -> &str {
        self.mapping.get_by_left(&idx).unwrap()
    }

    fn get_by_name(&self, name: &str) -> usize {
        self.get_by_name_opt(name).unwrap()
    }

    fn get_by_name_opt(&self, name: &str) -> Option<usize> {
        self.mapping.get_by_right(name).copied()
    }
}

impl<Ident: Display, Ptx: Display> Display for GenericInstruction<Ident, Ptx> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use GenericInstruction::*;
        match self {
            AddCtxCtx { out, in1, in2 } => write!(f, "{} = add {}, {}", out, in1, in2),
            AddPtxCtx { out, in1, in2 } => write!(f, "{} = add_ptx {}, {}", out, in1, in2),
            MulCtxCtx { out, in1, in2 } => write!(f, "{} = mul {}, {}", out, in1, in2),
            MulPtxCtx { out, in1, in2 } => write!(f, "{} = mul_ptx {}, {}", out, in1, in2),
            MulIntCtx { out, in1, in2 } => write!(f, "{} = mul_int {}, {}", out, in1, in2),
            Return { val } => write!(f, "return {}", val),
            Copy { out, in1 } => write!(f, "{} = copy {}", out, in1),
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
            InnerProduct { out, in1, in2 } => {
                write!(f, "{} = inner_prod ", out)?;
                for input in in1 {
                    write!(f, "{}, ", input)?;
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

impl<Ident: Display, Ptx: Display> Debug for GenericInstruction<Ident, Ptx> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl<Ident, Ptx> GenericInstruction<Ident, Ptx> {

    fn map_nonptx_identifiers<NewIdent, F>(self, f: &mut F) -> GenericInstruction<NewIdent, Ptx>
        where F: FnMut(Ident) -> NewIdent
    {
        use GenericInstruction::*;
        match self {
            AddCtxCtx { out, in1, in2 } => AddCtxCtx { out: f(out), in1: f(in1), in2: f(in2) },
            AddPtxCtx { out, in1, in2 } => AddPtxCtx { out: f(out), in1: f(in1), in2: in2 },
            MulCtxCtx { out, in1, in2 } => MulCtxCtx { out: f(out), in1: f(in1), in2: f(in2) },
            MulPtxCtx { out, in1, in2 } => MulPtxCtx { out: f(out), in1: f(in1), in2: in2 },
            MulIntCtx { out, in1, in2 } => MulIntCtx { out: f(out), in1: f(in1), in2: in2 },
            Zero { out } => Zero { out: f(out) },
            Copy { out, in1 } => Copy { out: f(out), in1: f(in1) },
            Galois { out, in1, exponents } => Galois { out: out.into_iter().map(&mut*f).collect(), in1: f(in1), exponents },
            Return { val } => Return { val: f(val) },
            InnerProduct { out, in1, in2 } => InnerProduct { out: f(out), in1: in1.into_iter().map(&mut*f).collect(), in2: in2 }
        }
    }

    fn map_ptx<NewPtx, F>(self, f: &mut F) -> GenericInstruction<Ident, NewPtx>
        where F: FnMut(Ptx) -> NewPtx
    {
        use GenericInstruction::*;
        match self {
            AddCtxCtx { out, in1, in2 } => AddCtxCtx { out: out, in1: in1, in2: in2 },
            AddPtxCtx { out, in1, in2 } => AddPtxCtx { out: out, in1: in1, in2: f(in2) },
            MulCtxCtx { out, in1, in2 } => MulCtxCtx { out: out, in1: in1, in2: in2 },
            MulPtxCtx { out, in1, in2 } => MulPtxCtx { out: out, in1: in1, in2: f(in2) },
            MulIntCtx { out, in1, in2 } => MulIntCtx { out: out, in1: in1, in2: in2 },
            Zero { out } => Zero { out: out },
            Copy { out, in1 } => Copy { out: out, in1: in1 },
            Galois { out, in1, exponents } => Galois { out: out, in1: in1, exponents },
            Return { val } => Return { val: val },
            InnerProduct { out, in1, in2 } => InnerProduct { out: out, in1: in1, in2: in2.into_iter().map(&mut*f).collect() }
        }
    }
}

impl<Ident> GenericInstruction<Ident> {

    fn map_identifiers<NewIdent, F>(self, f: &mut F) -> GenericInstruction<NewIdent>
        where F: FnMut(Ident) -> NewIdent
    {
        self.map_nonptx_identifiers(f).map_ptx(f)
    }
}

impl Display for PlaintextData {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

impl Debug for PlaintextData {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl From<Vec<i64>> for PlaintextData {

    fn from(value: Vec<i64>) -> Self {
        Self { data: value }
    }
}

impl Into<Vec<i64>> for PlaintextData {
    
    fn into(self) -> Vec<i64> {
        self.data
    }
}

impl Deref for PlaintextData {
    type Target = Vec<i64>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for PlaintextData {

    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<Ptx: Display> Display for Program<Ptx> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "func(")?;
        let mut params_it = self.inputs();
        if let Some(param) = params_it.next() {
            write!(f, "{}", param)?;
        }
        for param in params_it {
            write!(f, ", {}", param)?;
        }
        writeln!(f, ") {{")?;
        for inst in self.instructions() {
            writeln!(f, "    {}", inst)?;
        }
        writeln!(f, "}}")?;
        let mut plaintext_keys = self.plaintext_table.keys().map(|k| self.identifier_table.get(*k)).collect::<Vec<_>>();
        plaintext_keys.sort_unstable();
        for key in plaintext_keys {
            writeln!(f, "{}: {}", key, self.plaintext_table.get(&self.identifier_table.get_by_name(key)).unwrap())?;
        }
        return Ok(());
    }
}

impl<Ptx: Display> Debug for Program<Ptx> {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

fn parse_leading_identifier(data: &str, table: &mut IdentifierTable) -> Option<(usize, usize)> {
    if !(data.starts_with("%") || data.starts_with("@")) {
        return None;
    }
    let idx = data.chars().skip(1).take_while(|c| c.is_alphanumeric() || *c == '_').count();
    if idx > 0 {
        Some((table.ident_by_name(&data[..(idx + 1)]), idx + 1))
    } else {
        None
    }
}

fn expect<'a>(s: &mut &'a str, expected: &str) -> Option<()> {
    if s.starts_with(expected) {
        *s = &s[expected.len()..];
        return Some(());
    } else {
        return None;
    };
}

fn expect_ident<'a>(s: &mut &'a str, table: &mut IdentifierTable) -> Option<usize> {
    if let Some((ident, continue_at)) = parse_leading_identifier(*s, table) {
        *s = &s[continue_at..];
        return Some(ident);    
    } else {
        return None;
    };
}

fn expect_int<'a>(s: &mut &'a str) -> Option<i64> {
    let idx = s.chars().take_while(|c| *c == '-' || c.is_numeric()).count();
    if idx > 0 {
        let result = i64::from_str(&s[..idx]).ok();
        *s = &s[idx..];
        return result;
    } else {
        return None;
    }
}

fn expect_end<T>(s: &str, result: T) -> Option<T> {
    if s.len() == 0 {
        return Some(result);
    } else {
        return None;
    };
}

impl GenericInstruction<usize, usize> {

    fn parse(mut s: &str, table: &mut IdentifierTable) -> Option<Self> {
        s = s.trim();
        let mut outputs = Vec::new();
        if let Some(out) = expect_ident(&mut s, table) {
            outputs.push(out);
            while let Some(()) = expect(&mut s, ", ") {
                outputs.push(expect_ident(&mut s, table)?);
            }
        }
        let expect_single_output = || if outputs.len() == 1 {
            return Some(outputs[0]);
        } else {
            return None;
        };

        if let Some(()) = expect(&mut s, " = add ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s, table)?;
            expect_end(s, GenericInstruction::AddCtxCtx { out: expect_single_output()?, in1, in2 })
        } else if let Some(()) = expect(&mut s, " = add_ptx ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s, table)?;
            expect_end(s, GenericInstruction::AddPtxCtx { out: expect_single_output()?, in1, in2 })
        } else if let Some(()) = expect(&mut s, " = mul ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s, table)?;
            expect_end(s, GenericInstruction::MulCtxCtx { out: expect_single_output()?, in1, in2 })
        } else if let Some(()) = expect(&mut s, " = mul_ptx ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s, table)?;
            expect_end(s, GenericInstruction::MulPtxCtx { out: expect_single_output()?, in1, in2 })
        } else if let Some(()) = expect(&mut s, " = mul_int ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_int(&mut s)?;
            expect_end(s, GenericInstruction::MulIntCtx { out: expect_single_output()?, in1, in2 })
        } else if let Some(()) = expect(&mut s, " = galois ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", exponents = [")?;
            let mut exponents = Vec::new();
            if let Some(val) = expect_int(&mut s) {
                exponents.push(val);
                while let Some(()) = expect(&mut s, ", ") {
                    exponents.push(expect_int(&mut s)?);
                }
            }
            expect(&mut s, "]")?;
            if exponents.len() != outputs.len() || exponents.len() == 0 {
                return None;
            }
            expect_end(s, GenericInstruction::Galois { out: outputs, in1, exponents })
        } else if let Some(()) = expect(&mut s, " = copy ") {
            let val = expect_ident(&mut s, table)?;
            expect_end(s, GenericInstruction::Copy { out: expect_single_output()?, in1: val })
        } else if let Some(()) = expect(&mut s, "return ") {
            let val = expect_ident(&mut s, table)?;
            if outputs.len() == 0 {
                expect_end(s, GenericInstruction::Return { val })
            } else {
                None
            }
        } else if let Some(()) = expect(&mut s, " = inner_prod ") {
            let mut values = Vec::new();
            if let Some(val) = expect_ident(&mut s, table) {
                values.push(val);
                expect(&mut s, ", ")?;
                while let Some(val) = expect_ident(&mut s, table) {
                    values.push(val);
                    expect(&mut s, ", ")?;
                }
            }
            expect(&mut s, "coefficients = [")?;
            let mut coefficients = Vec::new();
            if let Some(coeff) = expect_ident(&mut s, table) {
                coefficients.push(coeff);
                while let Some(()) = expect(&mut s, ", ") {
                    coefficients.push(expect_ident(&mut s, table)?);
                }
            }
            expect(&mut s, "]")?;
            if coefficients.len() != values.len() {
                return None;
            }
            expect_end(s, GenericInstruction::InnerProduct { out: expect_single_output()?, in1: values, in2: coefficients })
        } else if let Some(()) = expect(&mut s, " = zero") {
            expect_end(s, GenericInstruction::Zero { out: expect_single_output()? })
        } else {
            None
        }
    }
}

impl<Ptx> Program<Ptx> {

    pub fn parse_impl<F: FnMut(&str) -> Option<Ptx>, R: Read>(data: R, mut parse_ptx: F) -> Result<Self, usize> {
        let mut ident_table = IdentifierTable { counter: 0, mapping: BiMap::new() };
        let mut lines = BufReader::new(data).lines().enumerate().map(|(num, line)| (num, line.unwrap())).filter(|(_, line)| line.trim() != "").fuse();
        let (first_line_num, first_line) = lines.next().ok_or(0usize)?;
        let mut s = first_line.as_str().trim();
        expect(&mut s, "func(").ok_or(first_line_num)?;
        let mut params = Vec::new();
        if let Some(param) = expect_ident(&mut s, &mut ident_table) {
            params.push(param);
            while let Some(()) = expect(&mut s, ", ") {
                params.push(expect_ident(&mut s, &mut ident_table).ok_or(first_line_num)?);
            }
        }
        expect(&mut s, ") {").ok_or(first_line_num)?;
        let mut instructions = Vec::new();
        let mut result = None;
        for (line_num, line) in lines.by_ref() {
            if let Some(instruction) = GenericInstruction::parse(line.as_str(), &mut ident_table) {
                instructions.push(instruction);
            } else if line.trim() == "}" {
                result = Some(Program {
                    identifier_table: ident_table,
                    instructions: instructions,
                    inputs: params,
                    plaintext_table: HashMap::new()
                });
                break;
            } else {
                Err(line_num)?;
            }
        }
        if result.is_none() {
            Err(usize::MAX)?;
        }
        let mut result = result.unwrap();
        for (line_num, line) in lines {
            let mut s = line.as_str().trim();
            let name = expect_ident(&mut s, &mut result.identifier_table).ok_or(line_num)?;
            expect(&mut s, ": ").ok_or(line_num)?;
            let data = parse_ptx(s).ok_or(line_num)?;
            result.plaintext_table.insert(name, data);
        }

        return Ok(result);
    }
}

#[cfg(test)]
use std::cell::RefCell;
#[cfg(test)]
use std::cmp::max;

#[test]
fn test_display_parse_no_data() {
    let actual: Program = Program::parse(r#"
        func(%x, %y) {
            %z = add %x, %y
            %a = inner_prod %x, %y, %z, coefficients = [@x, @y, @z]
            %b = zero
            %a = add %a, %b
            %a = mul_int %a, -5
            %c0, %c1 = galois %a, exponents = [5, -1]
            return %c0
            %c1 = add_ptx %c1, @c
            return %c1
        }
    "#.as_bytes()).unwrap();
    let expected = Program::new::<String, _>(&["%x", "%y"], [
        Instruction::AddCtxCtx { out: "%z", in1: "%x", in2: "%y" },
        Instruction::InnerProduct { out: "%a", in1: vec!["%x", "%y", "%z"], in2: vec!["@x", "@y", "@z"] },
        Instruction::Zero { out: "%b" },
        Instruction::AddCtxCtx { out: "%a", in1: "%a", in2: "%b" },
        Instruction::MulIntCtx { out: "%a", in1: "%a", in2: -5 },
        Instruction::Galois { out: vec!["%c0", "%c1"], in1: "%a", exponents: vec![5, -1] },
        Instruction::Return { val: "%c0" },
        Instruction::AddPtxCtx { out: "%c1", in1: "%c1", in2: "@c" },
        Instruction::Return { val: "%c1" }
    ], HashMap::new());
    assert_eq!(expected, actual);

    let actual = Program::parse(format!("{}", &expected).as_bytes()).unwrap();
    assert_eq!(expected, actual);

    let string = "func(%x, %y) {\n    %z0 = mul %x, %y\n    return %z0\n}\n";
    assert_eq!(string, format!("{}", <Program>::parse(string.as_bytes()).unwrap()));
}

#[test]
fn test_display_parse_with_data() {
    let actual: Program = Program::parse(r#"
        func(%x, %y) {
            %z = inner_prod %x, %y, coefficients = [@x, @y]
            return %z
        }
        @x: [1, 2, 3, 4, 5, 6, 7, 8]
        @y: [2, 3, 4, 5, 6, 7, 8, 9]
    "#.as_bytes()).unwrap();
    let expected = Program::new::<String, _>(&["%x", "%y"], [
        Instruction::InnerProduct { out: "%z", in1: vec!["%x", "%y"], in2: vec!["@x", "@y"] },
        Instruction::Return { val: "%z" }
    ], [
        ("@x".to_owned(), PlaintextData::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
        ("@y".to_owned(), PlaintextData::from(vec![2, 3, 4, 5, 6, 7, 8, 9]))
    ].into_iter().collect::<HashMap<_, _>>());
    assert_eq!(expected, actual);

    let actual = Program::parse(format!("{}", &expected).as_bytes()).unwrap();
    assert_eq!(expected, actual);

    let string = "func(%x) {\n    return %x\n}\n@x: [1, 2, 3, 4]\n";
    assert_eq!(string, format!("{}", <Program>::parse(string.as_bytes()).unwrap()));
}

#[test]
fn random_test_display_parse() {
    let rng = RefCell::new(oorandom::Rand64::new(0));
    let rand_usize = || [0, 1, 2, 4][usize::try_from(rng.borrow_mut().rand_u64() % 4).unwrap()];
    let rand_existing_ident: &dyn for<'a> Fn(&'a [String]) -> &'a str = &|existing_idents: &[String]| existing_idents[usize::try_from(rng.borrow_mut().rand_u64() % existing_idents.len() as u64).unwrap()].as_str();
    let rand_new_ident = |existing_idents: &mut Vec<String>| {
        let prefix = ["%", "@"][usize::try_from(rng.borrow_mut().rand_u64() % 2).unwrap()];
        let chars = ["a", "b", "c", "A", "B", "C", "_", "0", "1", "9"];
        let result = (0..(rand_usize() + 1)).map(|_| chars[usize::try_from(rng.borrow_mut().rand_u64() % chars.len() as u64).unwrap()]).fold(prefix.to_owned(), |current, next| current + next);
        existing_idents.push(result.clone());
        return result;
    };
    let rand_inst = |existing_idents: &mut Vec<String>| {
        let idx = rng.borrow_mut().rand_u64() % 10;
        match idx {
            0 => format!("{} = add {}, {}", rand_new_ident(existing_idents), rand_existing_ident(existing_idents), rand_existing_ident(existing_idents)),
            1 => format!("{} = add_ptx {}, {}", rand_new_ident(existing_idents), rand_existing_ident(existing_idents), rand_existing_ident(existing_idents)),
            2 => format!("{} = mul {}, {}", rand_new_ident(existing_idents), rand_existing_ident(existing_idents), rand_existing_ident(existing_idents)),
            3 => format!("{} = mul_ptx {}, {}", rand_new_ident(existing_idents), rand_existing_ident(existing_idents), rand_existing_ident(existing_idents)),
            4 => {
                let value = rng.borrow_mut().rand_i64();
                format!("{} = mul_int {}, {}", rand_new_ident(existing_idents), rand_existing_ident(existing_idents), value) 
            },
            5 => format!("{} = copy {}", rand_new_ident(existing_idents), rand_existing_ident(existing_idents)),
            6 => format!("{} = zero", rand_new_ident(existing_idents)),
            7 => format!("return {}", rand_existing_ident(existing_idents)),
            8 => {
                let count = max(rand_usize(), 1);
                let outputs = (0..count).map(|_| rand_new_ident(existing_idents)).reduce(|x, y| x + ", " + &y).unwrap_or("".to_owned());
                let exponents = (0..count).map(|_| format!("{}", rng.borrow_mut().rand_i64())).reduce(|x, y| x + ", " + &y).unwrap_or("".to_owned());
                format!("{} = galois {}, exponents = [{}]", outputs, rand_existing_ident(existing_idents), exponents)
            },
            9 => {
                let count = rand_usize();
                let values = (0..count).map(|_| format!("{}, ", rand_existing_ident(existing_idents))).reduce(|x, y| x + &y).unwrap_or("".to_owned());
                let coefficients = (0..count).map(|_| rand_new_ident(existing_idents)).reduce(|x, y| x + ", " + &y).unwrap_or("".to_owned());
                format!("{} = inner_prod {}coefficients = [{}]", rand_new_ident(existing_idents), values, coefficients)
            },
            10.. => unreachable!()
        }
    };
    for _ in 0..100 {
        let mut existing_idents = Vec::new();
        let inputs = (0..max(1, rand_usize())).map(|_| rand_new_ident(&mut existing_idents)).reduce(|x, y| x + ", " + &y).unwrap_or("".to_owned());
        let len = usize::try_from(rng.borrow_mut().rand_u64() % 20).unwrap();
        let string_repr = (0..len).map(|_| format!("    {}\n", rand_inst(&mut existing_idents))).fold("func(".to_owned() + &inputs + ") {\n", |x, y| x + &y) + "}\n";
        assert_eq!(string_repr, format!("{}", <Program>::parse(string_repr.as_bytes()).unwrap()));
    }
}