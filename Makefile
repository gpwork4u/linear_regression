target_file := linear_regression.py

all:
	python3 $(target_file)

check: all
	@diff -u test1 test2 && echo "OK" 
	@rm test*

