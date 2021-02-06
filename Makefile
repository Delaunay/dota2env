
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
	COVERAGE_FILE=.coverage.e2e coverage run --concurrency=multiprocessing tests/integrations/gym_tests.py

run-doctest:
	COVERAGE_FILE=.coverage.doct coverage run --parallel-mode -m pytest --cov=luafun --doctest-modules luafun

cov-combine:
	coverage combine
	coverage report -m
	coverage xml

cov-send: cov-combine
	codecov


install-lua:
	rm -f bots/botcpp_radiant.so
	rm -f bots/botcpp_dire.so
	ln -f botslua/bot_generic.lua bots/bot_generic.lua
	ln -f botslua/hero_selection.lua bots/hero_selection.lua
	ln -f botslua/ability_item_usage_generic.lua bots/ability_item_usage_generic.lua
	ln -f botslua/item_purchase_generic.lua bots/item_purchase_generic.lua
	ln -f botslua/pprint.lua bots/pprint.lua
	ln -f botscpp/bin/luafun.so bots/luafun.so

install-cpp:
	rm -f bots/bot_generic.lua
	rm -f bots/hero_selection.lua
	ln -f botscpp/bin/bots.so bots/botcpp_radiant.so
	ln -f botscpp/bin/bots.so bots/botcpp_dire.so

