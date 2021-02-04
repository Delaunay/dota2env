
rm-doc:
	rm -rf _build

build-doc:
	# sphinx-apidoc -e -o docs/api olympus
	sphinx-build --color -c docs/ -b html docs/ _build/html

serve-doc:
	sphinx-serve

update-doc: build-doc serve-doc

yolo: rm-doc build-doc serve-doc


run-1v1:
	coverage run tests/gym_tests.py

cov-combine:
	coverage combine
	coverage report -m
	coverage xml

cov-send: cov-combine
	codecov
