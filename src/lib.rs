#![warn(missing_docs)]
#![doc = include_str!("../Readme.md")]

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::io::{BufRead, BufReader, Read};
use std::ops::{Deref, DerefMut};
use std::str::FromStr;

use bimap::BiMap;

///
/// Wrapper around `Vec<i64>`, to be used to store plaintext elements.
///
#[derive(Clone, PartialEq, Eq)]
pub struct PlaintextData {
    data: Vec<i64>,
}

///
/// An FHE-IR instruction.
///
#[derive(Clone, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum GenericInstruction<Ident = usize, Ptx = Ident> {
    /// Ciphertext-ciphertext addition.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func(%input1, %input2) {
    ///         %output = add %input1, %input2
    ///         return %output
    ///     }
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    AddCtxCtx { out: Ident, lhs: Ident, rhs: Ident },
    /// Plaintext-ciphertext addition.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func(%input) {
    ///         %output = add_ptx %input, @constant
    ///         return %output
    ///     }
    ///     @constant: [1, 2, 3, 4]
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    AddPtxCtx {
        out: Ident,
        value: Ident,
        plaintext: Ptx,
    },
    /// Ciphertext-ciphertext multiplication.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func(%input1, %input2) {
    ///         %output = mul %input1, %input2
    ///         return %output
    ///     }
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    MulCtxCtx { out: Ident, lhs: Ident, rhs: Ident },
    /// Plaintext-ciphertext multiplication.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func(%input) {
    ///         %output = mul_ptx %input, @constant
    ///         return %output
    ///     }
    ///     @constant: [1, 2, 3, 4]
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    MulPtxCtx {
        out: Ident,
        value: Ident,
        plaintext: Ptx,
    },
    /// Integer-ciphertext multiplication.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func(%input) {
    ///         %output = mul_int %input, 42
    ///         return %output
    ///     }
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    MulIntCtx {
        out: Ident,
        value: Ident,
        integer: i64,
    },
    /// Create a copy of a ciphertext.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func(%input) {
    ///         %output = copy %input
    ///         return %output
    ///     }
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    Copy { out: Ident, val: Ident },
    /// Create a (transparent) zero ciphertext.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func() {
    ///         %output = zero
    ///         return %output
    ///     }
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    Zero { out: Ident },
    /// Return a value as output of the program.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func(%in_out_put) {
    ///         return %in_out_put
    ///     }
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    Return { val: Ident },
    /// Homomorphic Galois automorphism.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func(%input) {
    ///         %output1, %output2, %output3 = galois %input, exponents = [1, 5, -1]
    ///         return %output1
    ///         return %output2
    ///         return %output3
    ///     }
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    Galois {
        out: Vec<Ident>,
        val: Ident,
        exponents: Vec<i64>,
    },
    /// Plaintext-ciphertext inner product.
    ///
    /// # Example
    /// ```rust
    /// # use fhe_ir::*;
    /// <Program>::parse(r#"
    ///     func(%input1, %input2, %input3) {
    ///         %output = inner_prod %input1, %input2, %input3, coefficients = [@coeff1, @coeff2, @coeff3]
    ///         return %output
    ///     }
    ///     @coeff1: [2, 3]
    ///     @coeff2: [5, 7]
    ///     @coeff3: [4, 9]
    /// "#.as_bytes()).unwrap().check().unwrap();
    /// ```
    InnerProduct {
        out: Ident,
        values: Vec<Ident>,
        coefficients: Vec<Ptx>,
    },
}

///
/// An FHE-IR instruction without resolved constants.
///
pub type Instruction<'a> = GenericInstruction<&'a str, &'a str>;

///
/// An FHE-IR instruction with resolved constants, i.e. they contain a reference
/// to the data associated to each constant.
///
pub type InstructionWithData<'a, Ptx> = GenericInstruction<&'a str, &'a Ptx>;

///
/// An FHE-IR program, consisting of a list of inputs/parameters, a program body
/// formed by instructions, and a table mapping constants to their underlying
/// data.
///
#[derive(Clone, PartialEq, Eq)]
pub struct Program<Ptx = PlaintextData> {
    identifier_table: IdentifierTable,
    plaintext_table: HashMap<usize, Ptx>,
    inputs: Vec<usize>,
    instructions: Vec<GenericInstruction>,
}

impl<Ptx> Program<Ptx> {
    ///
    /// Creates a new [`Program`].
    ///
    /// The parameters are
    ///  - `inputs` is the list of identifiers that are used as inputs/parameters to the program
    ///  - `instructions` is the list of instructions that form the body of the program
    ///  - `plaintext_table` is maps the identifiers used for constants in the program
    ///    body to the actual data
    ///
    /// This does not perform any checks for semantic correctness. Use [`Program::new_check()`]
    /// as a shorthand for creating and checking the program.
    ///
    pub fn new<'a, K, I>(inputs: &[&str], instructions: I, plaintext_table: HashMap<K, Ptx>) -> Self
    where
        I: IntoIterator<Item = Instruction<'a>>,
        K: Borrow<str>,
    {
        let mut ident_table = IdentifierTable {
            counter: 0,
            mapping: BiMap::new(),
        };
        let inputs = inputs
            .iter()
            .map(|id| ident_table.ident_by_name(*id))
            .collect();
        let instructions = instructions
            .into_iter()
            .map(|inst| inst.map_identifiers(&mut |id| ident_table.ident_by_name(id)))
            .collect();
        let plaintext_table = plaintext_table
            .into_iter()
            .map(|(k, v)| (ident_table.ident_by_name(k.borrow()), v))
            .collect();
        return Self {
            identifier_table: ident_table,
            inputs: inputs,
            instructions: instructions,
            plaintext_table: plaintext_table,
        };
    }

    ///
    /// Shorthand for [`Program::new()`] followed by [`Program::check()`].
    ///
    pub fn new_check<'a, K, I>(
        inputs: &[&str],
        instructions: I,
        plaintext_table: HashMap<K, Ptx>,
    ) -> Result<Self, usize>
    where
        I: IntoIterator<Item = Instruction<'a>>,
        K: Borrow<str>,
    {
        let result = Program::new(inputs, instructions, plaintext_table);
        result.check()?;
        return Ok(result);
    }

    ///
    /// Returns the list of identifiers that are used as inputs/parameters to the program
    ///
    pub fn inputs<'a>(&'a self) -> impl ExactSizeIterator<Item = &'a str> {
        self.inputs
            .iter()
            .copied()
            .map(|idx| self.identifier_table.get(idx))
    }

    ///
    /// Returns the list of instructions that form the body of the program.
    ///
    /// This function does not resolve constants, but only returns their name.
    /// You can query the data behind the constant using [`Program::get_plaintext_data()`],
    /// or you can automatically resolve all constants using [`Program::instructions_with_data()`].
    ///
    pub fn instructions<'a>(&'a self) -> impl ExactSizeIterator<Item = Instruction<'a>> {
        self.instructions.iter().map(|inst| {
            inst.clone()
                .map_identifiers(&mut |idx| self.identifier_table.get(idx))
        })
    }

    ///
    /// Returns the list of instructions that form the body of the program.
    ///
    /// # Panics
    ///
    /// This function does resolve constants, so in particular will panic on invalid programs
    /// that don't have data available for every constant. If this is not desired, use
    /// [`Program::instructions()`] instead.
    ///
    pub fn instructions_with_data<'a>(
        &'a self,
    ) -> impl ExactSizeIterator<Item = InstructionWithData<'a, Ptx>> {
        self.instructions.iter().map(|inst| {
            inst.clone()
                .map_nonptx_identifiers(&mut |idx| self.identifier_table.get(idx))
                .map_ptx(&mut |key| self.plaintext_table.get(&key).unwrap())
        })
    }

    ///
    /// Returns an iterator over all constants and the associated values.
    ///
    /// The result is sorted lexicographically (w.r.t. the names of the constants).
    ///
    pub fn plaintext_table_entries<'a>(
        &'a self,
    ) -> impl ExactSizeIterator<Item = (&'a str, &'a Ptx)> {
        let mut plaintext_keys = self
            .plaintext_table
            .keys()
            .map(|k| self.identifier_table.get(*k))
            .collect::<Vec<_>>();
        plaintext_keys.sort_unstable();
        plaintext_keys.into_iter().map(|key| {
            (
                key,
                self.plaintext_table
                    .get(&self.identifier_table.get_by_name(key))
                    .unwrap(),
            )
        })
    }

    ///
    /// Returns the plaintext data associated with the given identifier, if any.
    ///
    pub fn get_plaintext_data<'a>(&'a self, identifier: &str) -> Option<&'a Ptx> {
        self.identifier_table
            .get_by_name_opt(identifier)
            .and_then(|idx| self.plaintext_table.get(&idx))
    }

    ///
    /// Checks whether this program is valid.
    ///
    /// In particular, this function checks the following:
    ///  - whether every used identifier has previously been initialized
    ///  - whether constant identifier start with `@` and other identifiers start with `%`
    ///  - whether every constant identifier is associated with underlying data
    ///  - whether the length of variadic inputs and outputs is valid and matches
    ///  - inputs/parameters are distinct
    ///
    /// In case of an error, the line number in which the error occured (assuming the program
    /// is formatted in the default way) is returned as [`Result::Err`].
    ///
    pub fn check(&self) -> Result<(), usize> {
        self.check_impl(|_| Ok(()))
    }

    ///
    /// Maps the data associated to each constant by using the given function.
    ///
    pub fn map_plaintexts<NewPtx, F>(self, mut f: F) -> Program<NewPtx>
    where
        F: FnMut(Ptx) -> NewPtx,
    {
        Program {
            identifier_table: self.identifier_table,
            inputs: self.inputs,
            instructions: self.instructions,
            plaintext_table: self
                .plaintext_table
                .into_iter()
                .map(|(k, v)| (k, f(v)))
                .collect(),
        }
    }
}

impl<Ptx: FromStr> Program<Ptx> {
    ///
    /// Parses a string into a [`Program`]. This is compatible with the [`Display`]
    /// implementation for [`Program`].
    ///
    /// This does not perform any checks beyond syntactic well-formedness.
    /// In case of a syntactic error, the line number of the error is returned as
    /// [`Result::Err`].
    ///
    pub fn parse<R: Read>(data: R) -> Result<Self, usize> {
        Self::parse_impl(data, |s| Ptx::from_str(s).map_err(|_| ()))
    }

    ///
    /// Shorthand for [`Program::parse()`] followed by [`Program::check()`].
    ///
    pub fn parse_check<R: Read>(data: R) -> Result<Self, usize> {
        let result = Self::parse(data)?;
        result.check()?;
        return Ok(result);
    }
}

impl GenericInstruction<usize, usize> {
    fn check<Ptx>(
        &self,
        defined_identifiers: &mut HashSet<usize>,
        identifier_table: &IdentifierTable,
        data_table: &HashMap<usize, Ptx>,
    ) -> Result<(), ()> {
        use GenericInstruction::*;
        let is_variable_name = |x: &usize| identifier_table.get(*x).starts_with("%");
        let is_constant_name = |x: &usize| identifier_table.get(*x).starts_with("@");
        let is_initialized = |x: &usize| defined_identifiers.contains(x);
        let has_data = |x: &usize| data_table.contains_key(x);
        if match self {
            AddCtxCtx {
                out,
                lhs: in1,
                rhs: in2,
            } => {
                is_variable_name(out)
                    && is_variable_name(in1)
                    && is_variable_name(in2)
                    && is_initialized(in1)
                    && is_initialized(in2)
            }
            AddPtxCtx {
                out,
                value: in1,
                plaintext: in2,
            } => {
                is_variable_name(out)
                    && is_variable_name(in1)
                    && is_constant_name(in2)
                    && is_initialized(in1)
                    && has_data(in2)
            }
            MulCtxCtx {
                out,
                lhs: in1,
                rhs: in2,
            } => {
                is_variable_name(out)
                    && is_variable_name(in1)
                    && is_variable_name(in2)
                    && is_initialized(in1)
                    && is_initialized(in2)
            }
            MulPtxCtx {
                out,
                value: in1,
                plaintext: in2,
            } => {
                is_variable_name(out)
                    && is_variable_name(in1)
                    && is_constant_name(in2)
                    && is_initialized(in1)
                    && has_data(in2)
            }
            MulIntCtx {
                out,
                value: in1,
                integer: _,
            } => is_variable_name(out) && is_variable_name(in1) && is_initialized(in1),
            Return { val } => is_variable_name(val) && is_initialized(val),
            Copy { out, val: in1 } => {
                is_variable_name(out) && is_variable_name(in1) && is_initialized(in1)
            }
            Zero { out } => is_variable_name(out),
            Galois {
                out,
                val: in1,
                exponents,
            } => {
                out.iter().all(|out| is_variable_name(out))
                    && is_variable_name(in1)
                    && is_initialized(in1)
                    && out.len() == exponents.len()
                    && exponents.len() > 0
            }
            InnerProduct {
                out,
                values: in1,
                coefficients: in2,
            } => {
                is_variable_name(out)
                    && in1
                        .iter()
                        .all(|in1| is_variable_name(in1) && is_initialized(in1))
                    && in2.iter().all(|in2| is_constant_name(in2) && has_data(in2))
                    && in1.len() == in2.len()
            }
        } {
            match self {
                AddCtxCtx {
                    out,
                    lhs: _,
                    rhs: _,
                }
                | AddPtxCtx {
                    out,
                    value: _,
                    plaintext: _,
                }
                | MulCtxCtx {
                    out,
                    lhs: _,
                    rhs: _,
                }
                | MulPtxCtx {
                    out,
                    value: _,
                    plaintext: _,
                }
                | MulIntCtx {
                    out,
                    value: _,
                    integer: _,
                }
                | Copy { out, val: _ }
                | Zero { out }
                | InnerProduct {
                    out,
                    values: _,
                    coefficients: _,
                } => {
                    defined_identifiers.insert(*out);
                }
                Galois {
                    out,
                    val: _,
                    exponents: _,
                } => {
                    defined_identifiers.extend(out.iter().copied());
                }
                Return { val: _ } => {}
            };
            Ok(())
        } else {
            Err(())
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct IdentifierTable {
    counter: usize,
    mapping: BiMap<usize, String>,
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
            AddCtxCtx {
                out,
                lhs: in1,
                rhs: in2,
            } => write!(f, "{} = add {}, {}", out, in1, in2),
            AddPtxCtx {
                out,
                value: in1,
                plaintext: in2,
            } => write!(f, "{} = add_ptx {}, {}", out, in1, in2),
            MulCtxCtx {
                out,
                lhs: in1,
                rhs: in2,
            } => write!(f, "{} = mul {}, {}", out, in1, in2),
            MulPtxCtx {
                out,
                value: in1,
                plaintext: in2,
            } => write!(f, "{} = mul_ptx {}, {}", out, in1, in2),
            MulIntCtx {
                out,
                value: in1,
                integer: in2,
            } => write!(f, "{} = mul_int {}, {}", out, in1, in2),
            Return { val } => write!(f, "return {}", val),
            Copy { out, val: in1 } => write!(f, "{} = copy {}", out, in1),
            Zero { out } => write!(f, "{} = zero", out),
            Galois {
                out,
                val: in1,
                exponents,
            } => {
                let mut out_it = out.iter();
                if let Some(out) = out_it.next() {
                    write!(f, "{}", out)?;
                }
                for out in out_it {
                    write!(f, ", {}", out)?;
                }
                write!(f, " = galois {}, exponents = {:?}", in1, exponents)
            }
            InnerProduct {
                out,
                values: in1,
                coefficients: in2,
            } => {
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
    where
        F: FnMut(Ident) -> NewIdent,
    {
        use GenericInstruction::*;
        match self {
            AddCtxCtx {
                out,
                lhs: in1,
                rhs: in2,
            } => AddCtxCtx {
                out: f(out),
                lhs: f(in1),
                rhs: f(in2),
            },
            AddPtxCtx {
                out,
                value: in1,
                plaintext: in2,
            } => AddPtxCtx {
                out: f(out),
                value: f(in1),
                plaintext: in2,
            },
            MulCtxCtx {
                out,
                lhs: in1,
                rhs: in2,
            } => MulCtxCtx {
                out: f(out),
                lhs: f(in1),
                rhs: f(in2),
            },
            MulPtxCtx {
                out,
                value: in1,
                plaintext: in2,
            } => MulPtxCtx {
                out: f(out),
                value: f(in1),
                plaintext: in2,
            },
            MulIntCtx {
                out,
                value: in1,
                integer: in2,
            } => MulIntCtx {
                out: f(out),
                value: f(in1),
                integer: in2,
            },
            Zero { out } => Zero { out: f(out) },
            Copy { out, val: in1 } => Copy {
                out: f(out),
                val: f(in1),
            },
            Galois {
                out,
                val: in1,
                exponents,
            } => Galois {
                out: out.into_iter().map(&mut *f).collect(),
                val: f(in1),
                exponents,
            },
            Return { val } => Return { val: f(val) },
            InnerProduct {
                out,
                values: in1,
                coefficients: in2,
            } => InnerProduct {
                out: f(out),
                values: in1.into_iter().map(&mut *f).collect(),
                coefficients: in2,
            },
        }
    }

    fn map_ptx<NewPtx, F>(self, f: &mut F) -> GenericInstruction<Ident, NewPtx>
    where
        F: FnMut(Ptx) -> NewPtx,
    {
        use GenericInstruction::*;
        match self {
            AddCtxCtx {
                out,
                lhs: in1,
                rhs: in2,
            } => AddCtxCtx {
                out: out,
                lhs: in1,
                rhs: in2,
            },
            AddPtxCtx {
                out,
                value: in1,
                plaintext: in2,
            } => AddPtxCtx {
                out: out,
                value: in1,
                plaintext: f(in2),
            },
            MulCtxCtx {
                out,
                lhs: in1,
                rhs: in2,
            } => MulCtxCtx {
                out: out,
                lhs: in1,
                rhs: in2,
            },
            MulPtxCtx {
                out,
                value: in1,
                plaintext: in2,
            } => MulPtxCtx {
                out: out,
                value: in1,
                plaintext: f(in2),
            },
            MulIntCtx {
                out,
                value: in1,
                integer: in2,
            } => MulIntCtx {
                out: out,
                value: in1,
                integer: in2,
            },
            Zero { out } => Zero { out: out },
            Copy { out, val: in1 } => Copy { out: out, val: in1 },
            Galois {
                out,
                val: in1,
                exponents,
            } => Galois {
                out: out,
                val: in1,
                exponents,
            },
            Return { val } => Return { val: val },
            InnerProduct {
                out,
                values: in1,
                coefficients: in2,
            } => InnerProduct {
                out: out,
                values: in1,
                coefficients: in2.into_iter().map(&mut *f).collect(),
            },
        }
    }
}

impl<Ident> GenericInstruction<Ident> {
    fn map_identifiers<NewIdent, F>(self, f: &mut F) -> GenericInstruction<NewIdent>
    where
        F: FnMut(Ident) -> NewIdent,
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

impl From<PlaintextData> for Vec<i64> {
    fn from(value: PlaintextData) -> Vec<i64> {
        value.data
    }
}

impl FromStr for PlaintextData {

    type Err = ();
    
    fn from_str(mut s: &str) -> Result<Self, Self::Err> {
        let mut data = Vec::new();
        expect(&mut s, "[").ok_or(())?;
        if let Some(val) = expect_int(&mut s) {
            data.push(val);
            while let Some(()) = expect(&mut s, ", ") {
                data.push(expect_int(&mut s).ok_or(())?);
            }
        }
        expect(&mut s, "]").ok_or(())?;
        expect_end(s, PlaintextData::from(data)).ok_or(())
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
        for (key, val) in self.plaintext_table_entries() {
            writeln!(f, "{}: {}", key, val)?;
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
    let idx = data
        .chars()
        .skip(1)
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .count();
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
    let idx = s
        .chars()
        .take_while(|c| *c == '-' || c.is_ascii_digit())
        .count();
    if idx > 0 {
        let result = i64::from_str(&s[..idx]).ok()?;
        *s = &s[idx..];
        return Some(result);
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
        let expect_single_output = || {
            if outputs.len() == 1 {
                return Some(outputs[0]);
            } else {
                return None;
            }
        };

        if let Some(()) = expect(&mut s, " = add ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s, table)?;
            expect_end(
                s,
                GenericInstruction::AddCtxCtx {
                    out: expect_single_output()?,
                    lhs: in1,
                    rhs: in2,
                },
            )
        } else if let Some(()) = expect(&mut s, " = add_ptx ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s, table)?;
            expect_end(
                s,
                GenericInstruction::AddPtxCtx {
                    out: expect_single_output()?,
                    value: in1,
                    plaintext: in2,
                },
            )
        } else if let Some(()) = expect(&mut s, " = mul ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s, table)?;
            expect_end(
                s,
                GenericInstruction::MulCtxCtx {
                    out: expect_single_output()?,
                    lhs: in1,
                    rhs: in2,
                },
            )
        } else if let Some(()) = expect(&mut s, " = mul_ptx ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_ident(&mut s, table)?;
            expect_end(
                s,
                GenericInstruction::MulPtxCtx {
                    out: expect_single_output()?,
                    value: in1,
                    plaintext: in2,
                },
            )
        } else if let Some(()) = expect(&mut s, " = mul_int ") {
            let in1 = expect_ident(&mut s, table)?;
            expect(&mut s, ", ")?;
            let in2 = expect_int(&mut s)?;
            expect_end(
                s,
                GenericInstruction::MulIntCtx {
                    out: expect_single_output()?,
                    value: in1,
                    integer: in2,
                },
            )
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
            expect_end(
                s,
                GenericInstruction::Galois {
                    out: outputs,
                    val: in1,
                    exponents,
                },
            )
        } else if let Some(()) = expect(&mut s, " = copy ") {
            let val = expect_ident(&mut s, table)?;
            expect_end(
                s,
                GenericInstruction::Copy {
                    out: expect_single_output()?,
                    val,
                },
            )
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
            expect_end(
                s,
                GenericInstruction::InnerProduct {
                    out: expect_single_output()?,
                    values,
                    coefficients,
                },
            )
        } else if let Some(()) = expect(&mut s, " = zero") {
            expect_end(
                s,
                GenericInstruction::Zero {
                    out: expect_single_output()?,
                },
            )
        } else {
            None
        }
    }
}

impl<Ptx> Program<Ptx> {
    fn parse_impl<F: FnMut(&str) -> Result<Ptx, ()>, R: Read>(
        data: R,
        mut parse_ptx: F,
    ) -> Result<Self, usize> {
        let mut ident_table = IdentifierTable {
            counter: 0,
            mapping: BiMap::new(),
        };
        let mut lines = BufReader::new(data)
            .lines()
            .enumerate()
            .map(|(num, line)| (num, line.unwrap()))
            .filter(|(_, line)| line.trim() != "")
            .fuse();
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
        expect_end(s, ()).ok_or(first_line_num)?;

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
                    plaintext_table: HashMap::new(),
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
            let data = parse_ptx(s).map_err(|()| line_num)?;
            if result.plaintext_table.contains_key(&name) {
                Err(line_num)?;
            } else {
                result.plaintext_table.insert(name, data);
            }
        }

        return Ok(result);
    }

    fn check_impl<F>(&self, mut check_ptx: F) -> Result<(), usize>
    where
        F: FnMut(&Ptx) -> Result<(), ()>,
    {
        let mut initialized_identifiers = HashSet::new();
        let mut num_offset = 0;
        for input in self.inputs.iter() {
            if !self.identifier_table.get(*input).starts_with("%") {
                Err(num_offset)?;
            } else if initialized_identifiers.contains(input) {
                Err(num_offset)?;
            } else {
                initialized_identifiers.insert(*input);
            }
        }
        num_offset += 1;
        for (num, inst) in self.instructions.iter().enumerate() {
            inst.check(
                &mut initialized_identifiers,
                &self.identifier_table,
                &self.plaintext_table,
            )
            .map_err(|()| num + num_offset)?;
        }
        num_offset += self.instructions.len();
        for (num, (constant, val)) in self.plaintext_table_entries().enumerate() {
            check_ptx(val).map_err(|()| num + num_offset)?;
            if !constant.starts_with("@") {
                Err(num + num_offset)?;
            }
        }
        return Ok(());
    }
}

#[cfg(test)]
use std::cell::RefCell;
#[cfg(test)]
use std::cmp::max;

#[test]
fn test_display_parse_no_data() {
    let actual: Program = Program::parse(
        r#"
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
    "#
        .as_bytes(),
    )
    .unwrap();
    let expected = Program::new::<String, _>(
        &["%x", "%y"],
        [
            Instruction::AddCtxCtx {
                out: "%z",
                lhs: "%x",
                rhs: "%y",
            },
            Instruction::InnerProduct {
                out: "%a",
                values: vec!["%x", "%y", "%z"],
                coefficients: vec!["@x", "@y", "@z"],
            },
            Instruction::Zero { out: "%b" },
            Instruction::AddCtxCtx {
                out: "%a",
                lhs: "%a",
                rhs: "%b",
            },
            Instruction::MulIntCtx {
                out: "%a",
                value: "%a",
                integer: -5,
            },
            Instruction::Galois {
                out: vec!["%c0", "%c1"],
                val: "%a",
                exponents: vec![5, -1],
            },
            Instruction::Return { val: "%c0" },
            Instruction::AddPtxCtx {
                out: "%c1",
                value: "%c1",
                plaintext: "@c",
            },
            Instruction::Return { val: "%c1" },
        ],
        HashMap::new(),
    );
    assert_eq!(expected, actual);

    let actual = Program::parse(format!("{}", &expected).as_bytes()).unwrap();
    assert_eq!(expected, actual);

    let string = "func(%x, %y) {\n    %z0 = mul %x, %y\n    return %z0\n}\n";
    assert_eq!(
        string,
        format!("{}", <Program>::parse(string.as_bytes()).unwrap())
    );
}

#[test]
fn test_display_parse_with_data() {
    let actual: Program = Program::parse(
        r#"
        func(%x, %y) {
            %z = inner_prod %x, %y, coefficients = [@x, @y]
            return %z
        }
        @x: [1, 2, 3, 4, 5, 6, 7, 8]
        @y: [2, 3, 4, 5, 6, 7, 8, 9]
    "#
        .as_bytes(),
    )
    .unwrap();
    let expected = Program::new::<String, _>(
        &["%x", "%y"],
        [
            Instruction::InnerProduct {
                out: "%z",
                values: vec!["%x", "%y"],
                coefficients: vec!["@x", "@y"],
            },
            Instruction::Return { val: "%z" },
        ],
        [
            (
                "@x".to_owned(),
                PlaintextData::from(vec![1, 2, 3, 4, 5, 6, 7, 8]),
            ),
            (
                "@y".to_owned(),
                PlaintextData::from(vec![2, 3, 4, 5, 6, 7, 8, 9]),
            ),
        ]
        .into_iter()
        .collect::<HashMap<_, _>>(),
    );
    assert_eq!(expected, actual);

    let actual = Program::parse(format!("{}", &expected).as_bytes()).unwrap();
    assert_eq!(expected, actual);

    let string = "func(%x) {\n    return %x\n}\n@x: [1, 2, 3, 4]\n";
    assert_eq!(
        string,
        format!("{}", <Program>::parse(string.as_bytes()).unwrap())
    );
}

#[test]
fn random_test_display_parse() {
    let rng = RefCell::new(oorandom::Rand64::new(0));
    let rand_usize = || [0, 1, 2, 4][usize::try_from(rng.borrow_mut().rand_u64() % 4).unwrap()];
    let rand_existing_ident: &dyn for<'a> Fn(&'a [String]) -> &'a str =
        &|existing_idents: &[String]| {
            existing_idents[usize::try_from(
                rng.borrow_mut().rand_u64() % existing_idents.len() as u64,
            )
            .unwrap()]
            .as_str()
        };
    let rand_new_ident = |existing_idents: &mut Vec<String>| {
        let prefix = ["%", "@"][usize::try_from(rng.borrow_mut().rand_u64() % 2).unwrap()];
        let chars = ["a", "b", "c", "A", "B", "C", "_", "0", "1", "9"];
        let result = (0..(rand_usize() + 1))
            .map(|_| {
                chars[usize::try_from(rng.borrow_mut().rand_u64() % chars.len() as u64).unwrap()]
            })
            .fold(prefix.to_owned(), |current, next| current + next);
        existing_idents.push(result.clone());
        return result;
    };
    let rand_inst = |existing_idents: &mut Vec<String>| {
        let idx = rng.borrow_mut().rand_u64() % 10;
        match idx {
            0 => format!(
                "{} = add {}, {}",
                rand_new_ident(existing_idents),
                rand_existing_ident(existing_idents),
                rand_existing_ident(existing_idents)
            ),
            1 => format!(
                "{} = add_ptx {}, {}",
                rand_new_ident(existing_idents),
                rand_existing_ident(existing_idents),
                rand_existing_ident(existing_idents)
            ),
            2 => format!(
                "{} = mul {}, {}",
                rand_new_ident(existing_idents),
                rand_existing_ident(existing_idents),
                rand_existing_ident(existing_idents)
            ),
            3 => format!(
                "{} = mul_ptx {}, {}",
                rand_new_ident(existing_idents),
                rand_existing_ident(existing_idents),
                rand_existing_ident(existing_idents)
            ),
            4 => {
                let value = rng.borrow_mut().rand_i64();
                format!(
                    "{} = mul_int {}, {}",
                    rand_new_ident(existing_idents),
                    rand_existing_ident(existing_idents),
                    value
                )
            }
            5 => format!(
                "{} = copy {}",
                rand_new_ident(existing_idents),
                rand_existing_ident(existing_idents)
            ),
            6 => format!("{} = zero", rand_new_ident(existing_idents)),
            7 => format!("return {}", rand_existing_ident(existing_idents)),
            8 => {
                let count = max(rand_usize(), 1);
                let outputs = (0..count)
                    .map(|_| rand_new_ident(existing_idents))
                    .reduce(|x, y| x + ", " + &y)
                    .unwrap_or("".to_owned());
                let exponents = (0..count)
                    .map(|_| format!("{}", rng.borrow_mut().rand_i64()))
                    .reduce(|x, y| x + ", " + &y)
                    .unwrap_or("".to_owned());
                format!(
                    "{} = galois {}, exponents = [{}]",
                    outputs,
                    rand_existing_ident(existing_idents),
                    exponents
                )
            }
            9 => {
                let count = rand_usize();
                let values = (0..count)
                    .map(|_| format!("{}, ", rand_existing_ident(existing_idents)))
                    .reduce(|x, y| x + &y)
                    .unwrap_or("".to_owned());
                let coefficients = (0..count)
                    .map(|_| rand_new_ident(existing_idents))
                    .reduce(|x, y| x + ", " + &y)
                    .unwrap_or("".to_owned());
                format!(
                    "{} = inner_prod {}coefficients = [{}]",
                    rand_new_ident(existing_idents),
                    values,
                    coefficients
                )
            }
            10.. => unreachable!(),
        }
    };
    for _ in 0..100 {
        let mut existing_idents = Vec::new();
        let inputs = (0..max(1, rand_usize()))
            .map(|_| rand_new_ident(&mut existing_idents))
            .reduce(|x, y| x + ", " + &y)
            .unwrap_or("".to_owned());
        let len = usize::try_from(rng.borrow_mut().rand_u64() % 20).unwrap();
        let string_repr = (0..len)
            .map(|_| format!("    {}\n", rand_inst(&mut existing_idents)))
            .fold("func(".to_owned() + &inputs + ") {\n", |x, y| x + &y)
            + "}\n";
        assert_eq!(
            string_repr,
            format!("{}", <Program>::parse(string_repr.as_bytes()).unwrap())
        );
    }
}

#[test]
fn test_check() {
    let actual: Program = Program::parse(
        r#"
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
        @x: [1]
        @y: [2]
        @z: [3]
        @c: [4]
    "#
        .as_bytes(),
    )
    .unwrap();
    actual.check().unwrap();
}
