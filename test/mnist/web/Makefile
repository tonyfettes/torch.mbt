TARGETS := wasm wasm-gc
PROFILES := debug release
MOON_FLAGS :=

default: main

define make_moon_rules
ifeq ($(2),debug)
target/$(1)/$(2)/build/$(3).wasm:
	moon build --target $(1) --debug $(MOON_FLAGS)
else
target/$(1)/$(2)/build/$(3).wasm:
	moon build --target $(1) $(MOON_FLAGS)
endif
endef

$(foreach target,$(TARGETS),$(foreach profile,$(PROFILES),$(eval $(call make_moon_rules,$(target),$(profile),gen/gen))))

WIT  := wit

TARGET := wasm
PROFILE := debug
BUILD := target/$(TARGET)/$(PROFILE)/build

ifeq ($(TARGET),wasm)
ifeq ($(PROFILE),release)
main: $(BUILD)/gen/gen.opt.component.wasm
	node_modules/.bin/jco transpile $< -o src/ --name mnist
else
main: $(BUILD)/gen/gen.component.wasm
	node_modules/.bin/jco transpile $< -o src/ --name mnist
endif
else ifeq ($(TARGET),wasm-gc)
main: target/wasm/$(PROFILE)/build/gen/gen.component.wasm target/wasm-gc/$(PROFILE)/build/gen/gen.wasm
	node_modules/.bin/jco transpile $< -o src/ --name mnist
	cp target/wasm-gc/$(PROFILE)/build/gen/gen.wasm src/mnist.core.wasm
endif

%.wat: %.wasm
	wasm-tools print $< > $@

%.opt.wasm: %.wasm
	wasm-opt --enable-gc --enable-multivalue --enable-reference-types --enable-bulk-memory $< -O3 -o $@

%.embedded.wasm: $(WIT) %.wasm
	wasm-tools component embed $(WIT) $*.wasm -o $@ --encoding utf16

%.component.wasm: %.embedded.wasm
	wasm-tools component new $< -o $@

clean:
	moon clean

serve: main
	pnpm exec serve -s src/

.PHONY: main clean $(foreach target,$(TARGETS),$(foreach profile,$(PROFILES),target/$(target)/$(profile)/build/gen/gen.wasm))
