TARGETS := wasm wasm-gc native js
PROFILES := debug release
MOON_FLAGS :=
PACKAGE := gen/gen

default: src/mnist/model

define make_moon_rules
ifeq ($(2),debug)
.PHONY: target/$(1)/$(2)/build/$(PACKAGE).wasm
target/$(1)/$(2)/build/$(PACKAGE).wasm:
	moon build --target $(1) --debug $(MOON_FLAGS)
else
.PHONY: target/$(1)/$(2)/build/$(PACKAGE).wasm
target/$(1)/$(2)/build/$(PACKAGE).wasm:
	moon build --target $(1) $(MOON_FLAGS)
endif
endef

$(foreach target,$(TARGETS),$(foreach profile,$(PROFILES),$(eval $(call make_moon_rules,$(target),$(profile)))))

WIT = wit

TARGET = wasm
PROFILE = debug
BUILD := target/$(TARGET)/$(PROFILE)/build

%.wat: %.wasm
	wasm-tools print $< > $@

%.opt.wasm: %.wasm
	wasm-opt --enable-gc --enable-multivalue --enable-reference-types --enable-bulk-memory $< -O3 -o $@

%.embedded.wasm: $(WIT) %.wasm
	wasm-tools component embed $(WIT) $*.wasm -o $@

%.component.wasm: %.embedded.wasm
	wasm-tools component new $< -o $@

src/mnist/model: $(BUILD)/$(PACKAGE).component.wasm
	python3 -m wasmtime.bindgen $< --out-dir $@

run: main main.py
	python3 main.py

wit-bindgen:
	wit-bindgen moonbit wit --derive-show --derive-eq --derive-error --ignore-stub --out-dir .

.PHONY: default
