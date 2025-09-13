all: hello.exe

%.exe: %.rs
	rustc $< -o $@

clean:
	rm -f *.exe
