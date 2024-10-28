WIT  := wit

TARGETS := wasm wasm-gc
PROFILES := debug release
MOON_FLAGS :=
PACKAGE := gen/gen

TARGET := wasm
PROFILE := debug
BUILD := target/$(TARGET)/$(PROFILE)/build

WASM_OPT_FLAGS := --enable-gc --enable-multivalue --enable-reference-types --enable-bulk-memory --enable-simd

JCO_OUT := web
JCO_NAME := mnist-model
JCO := corepack pnpm exec jco
JCO_FLAGS := --no-nodejs-compat
JCO_OUTPUTS := $(JCO_OUT)/$(JCO_NAME).core.wasm $(JCO_OUT)/$(JCO_NAME).d.ts $(JCO_OUT)/$(JCO_NAME).js

PY_OUT := cli
PY_NAME := mnist/model

default: $(JCO_OUTPUTS)

define make_moon_rules
ifeq ($(2),debug)
target/$(1)/$(2)/build/$(3).wasm:
	moon build --target $(1) --debug $(MOON_FLAGS)
else
target/$(1)/$(2)/build/$(3).wasm:
	moon build --target $(1) $(MOON_FLAGS)
endif
endef

$(foreach target,$(TARGETS),$(foreach profile,$(PROFILES),$(eval $(call make_moon_rules,$(target),$(profile),$(PACKAGE)))))

ifeq ($(TARGET),wasm)
ifeq ($(PROFILE),release)
$(JCO_OUTPUTS) &: $(BUILD)/$(PACKAGE).opt.utf16.component.wasm
	$(JCO) transpile $(JCO_FLAGS) $< -o $(JCO_OUT)/ --name $(JCO_NAME)
else
$(JCO_OUTPUTS) &: $(BUILD)/$(PACKAGE).utf16.component.wasm
	$(JCO) transpile $(JCO_FLAGS) $< -o $(JCO_OUT)/ --name $(JCO_NAME)
endif
else ifeq ($(TARGET),wasm-gc)
$(JCO_OUTPUTS) &: target/wasm/$(PROFILE)/build/$(PACKAGE).utf16.component.wasm | target/wasm-gc/$(PROFILE)/build/$(PACKAGE).wasm
	$(JCO) transpile $(JCO_FLAGS) $< -o $(JCO_OUT)/ --name $(JCO_NAME)
	cp target/wasm-gc/$(PROFILE)/build/$(PACKAGE).wasm $(JCO_OUT)/$(JCO_NAME).core.wasm
endif

$(PY_OUT)/$(PY_NAME): target/wasm/$(PROFILE)/build/$(PACKAGE).utf8.component.wasm
	python3 -m wasmtime.bindgen $< --out-dir $@

%.wat: %.wasm
	wasm-tools print $< > $@

%.opt.wasm: %.wasm
	wasm-opt $(WASM_OPT_FLAGS) $< -O3 -o $@

%.utf8.embedded.wasm: $(WIT) %.wasm
	wasm-tools component embed $(WIT) $*.wasm -o $@

%.utf16.embedded.wasm: $(WIT) %.wasm
	wasm-tools component embed $(WIT) $*.wasm -o $@ --encoding utf16

%.component.wasm: %.embedded.wasm
	wasm-tools component new $< -o $@

wit-bindgen:
	wit-bindgen moonbit wit --derive-show --derive-eq --derive-error --ignore-stub --out-dir src

clean:
	moon clean

serve:
	corepack pnpm exec serve -c ./serve.json

.PHONY: default clean $(foreach target,$(TARGETS),$(foreach profile,$(PROFILES),target/$(target)/$(profile)/build/$(PACKAGE).wasm))
