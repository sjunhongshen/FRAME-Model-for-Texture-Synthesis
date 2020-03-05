all: libjulesz.so

libjulesz.so: julesz.c
	$(CC) -Wall -g -fPIC -shared -o $@ $? -lc

clean:
	rm -f test libjulesz.o libjulesz.so *.pyc
	rm -rf build