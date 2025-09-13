#[derive(Debug)]
struct NamedConstant<'a> {
    name: &'a str,
    value: f64,
}

fn main() {
    let arr = [1, 2, 3];
    println!("{:?}... Let's go!", arr);

    let pi = NamedConstant {
        name: "pi",
        value: 3.1415,
    };
    println!("I am a named constant: {:#?}", pi);
}
