# A minimalistic, scheme-agnostic IR for FHE

This library models a minimalistic, scheme-agnostic Intermediate Representation (IR) for Fully Homomorphic Encryption (FHE), with a simple parser and formatter.
Its goal is to provide an easy and debuggable way to exchange small circuits, consisting of arithmetic and Galois gates, between programs and functions.
The main use case we had in mind when designing this is to model and store circuits for the linear transforms and digit extraction during BFV/BGV/GBFV/dBFV bootstrapping.
This is not an attempt to replicate HEIR or other FHE IR projects, and is (intentionally) extremely limited in scope. 

Features:
 - Modelling of an IR for FHE circuits, including homomorphic addition, multiplication, galois automorphisms and inner products
 - MLIR-inspired (but not fully compatible) syntax
 - Parsing and formatting of such programs

Non-Features:
 - It is not a compiler, doesn't support any optimization passes, and only very rudimentary type checking
 - It doesn't have any scheme-specific operations like modulus-switching, exact division or plaintext-as-ciphertext reinterpretation
 - It doesn't contain an interpreter
 - It doesn't know about finite fields, and stores data as integers; host code will have to ensure that programs are generated and used with the same rings and FHE parameters
 - While MLIR-inspired, this doesn't use SSA; An instruction like `%x = add %x, %y` is allowed and intended

# Example

```rust
# use fhe_ir::*;
let program = Program::parse_check(r#"
    func(%x) {
        %x1, %x2, %x3, %x4 = galois %x, exponents = [1, 5, -1, -5]
        %result = inner_prod %x1, %x2, %x3, %x4, coefficients = [@c1, @c2, @c3, @c4]
        return %result
    }
    @c1: []
    @c2: []
    @c3: []
    @c4: []
"#.as_bytes()).unwrap();

for inst in program.instructions_with_data() {
    if let InstructionWithData::InnerProduct { out, values, coefficients } = inst {
        println!("Compute inner product of {:?} and {:?}", values, coefficients);
    }
}
```