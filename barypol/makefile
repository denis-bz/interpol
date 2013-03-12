# barypol/cython/makefile

Src = *.pyx setup.py h/*.h
Srcall = $(Src) test-*.py makefile

barypol.so: $(Src)
	-@ rm $@ 2> /dev/null || true
	setup.py  # cython .pyx ... > barypol.so
# cd ..; ln -s cython/barypol.so .


ls::
	@ ls -1 $(Srcall)
ll::
	@ ll -t $(Srcall)

day = $(shell day)

Tar = barypol-$(day).tar.gz
tar::
	tar -cz $(Srcall)  > $(Tar)
	tarlis $(Tar)

Bak = .bak-$(day)
bak::
	- rm -rf $(Bak);  mkdir $(Bak);  cp -p $(Srcall) $(Bak)

clean::
	- rm -rf a.out barypol.cpp build/

.SUFFIXES:
