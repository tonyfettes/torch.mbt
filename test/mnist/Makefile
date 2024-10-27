TARGETS := wasm wasm-gc
PROFILES := debug release
MOON_FLAGS :=
PACKAGE := gen/gen
OUT := scripts
NAME := mnist-model
JCO := corepack pnpm exec jco
JCO_FLAGS := --no-nodejs-compat
WASM_OPT_FLAGS := --enable-gc --enable-multivalue --enable-reference-types --enable-bulk-memory --enable-simd

JCO_OUTPUTS := $(OUT)/$(NAME).core.wasm $(OUT)/$(NAME).d.ts $(OUT)/$(NAME).js

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

WIT  := wit

TARGET := wasm
PROFILE := debug
BUILD := target/$(TARGET)/$(PROFILE)/build

ifeq ($(TARGET),wasm)
ifeq ($(PROFILE),release)
$(JCO_OUTPUTS) &: $(BUILD)/$(PACKAGE).opt.component.wasm
	$(JCO) transpile $(JCO_FLAGS) $< -o $(OUT)/ --name $(NAME)
else
$(JCO_OUTPUTS) &: $(BUILD)/$(PACKAGE).component.wasm
	$(JCO) transpile $(JCO_FLAGS) $< -o $(OUT)/ --name $(NAME)
endif
else ifeq ($(TARGET),wasm-gc)
$(JCO_OUTPUTS) &: target/wasm/$(PROFILE)/build/$(PACKAGE).component.wasm target/wasm-gc/$(PROFILE)/build/$(PACKAGE).wasm
	$(JCO) transpile $(JCO_FLAGS) $< -o $(OUT)/ --name $(NAME)
	cp target/wasm-gc/$(PROFILE)/build/$(PACKAGE).wasm $(OUT)/$(NAME).core.wasm
endif

%.wat: %.wasm
	wasm-tools print $< > $@

%.opt.wasm: %.wasm
	wasm-opt $(WASM_OPT_FLAGS) $< -O3 -o $@

%.embedded.wasm: $(WIT) %.wasm
	wasm-tools component embed $(WIT) $*.wasm -o $@ --encoding utf16

%.component.wasm: %.embedded.wasm
	wasm-tools component new $< -o $@

wit-bindgen:
	wit-bindgen moonbit wit --derive-show --derive-eq --derive-error --ignore-stub --out-dir .

clean:
	moon clean

serve:
	corepack pnpm exec serve -c ./serve.json

.PHONY: default clean $(foreach target,$(TARGETS),$(foreach profile,$(PROFILES),target/$(target)/$(profile)/build/$(PACKAGE).wasm))
