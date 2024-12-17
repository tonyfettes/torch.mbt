///|
let source =
  #|const isNode = typeof process !== "undefined";
  #|
  #|let onMessage;
  #|if (navigator.gpu) {
  #|  const adapter = await navigator.gpu.requestAdapter();
  #|
  #|  if (!adapter) {
  #|    throw new Error("Couldn't request WebGPU adapter.");
  #|  }
  #|
  #|  const device = await adapter.requestDevice();
  #|
  #|  onMessage = async (event) => {
  #|    const { buffer } = event;
  #|    const controlLength = 2;
  #|    const control = new Int32Array(buffer, 0, controlLength);
  #|    const length = control[1];
  #|    const inputA = new Float32Array(buffer, controlLength * 4, length);
  #|    const inputB = new Float32Array(
  #|      buffer,
  #|      controlLength * 4 + inputA.byteLength,
  #|      length
  #|    );
  #|    const inputABuffer = device.createBuffer({
  #|      size: length * 4,
  #|      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  #|      mappedAtCreation: true,
  #|    });
  #|    new Float32Array(inputABuffer.getMappedRange()).set(inputA);
  #|    const inputBBuffer = device.createBuffer({
  #|      size: length * 4,
  #|      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  #|      mappedAtCreation: true,
  #|    });
  #|    new Float32Array(inputBBuffer.getMappedRange()).set(inputB);
  #|    inputABuffer.unmap();
  #|    inputBBuffer.unmap();
  #|    const outputBuffer = device.createBuffer({
  #|      size: length * 4,
  #|      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  #|    });
  #|    const stagingBuffer = device.createBuffer({
  #|      size: length * 4,
  #|      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  #|    });
  #|    const shader = `
  #|      @group(0) @binding(0)
  #|      var<storage, read> inputA: array<f32>;
  #|
  #|      @group(0) @binding(1)
  #|      var<storage, read> inputB: array<f32>;
  #|
  #|      @group(0) @binding(2)
  #|      var<storage, read_write> output: array<f32>;
  #|
  #|      @compute @workgroup_size(64)
  #|      fn main(
  #|        @builtin(global_invocation_id)
  #|        global_id : vec3u,
  #|      ) {
  #|        if (global_id.x >= ${inputA.length}) {
  #|          return;
  #|        }
  #|
  #|        output[global_id.x] = inputA[global_id.x] + inputB[global_id.x];
  #|      }`;
  #|    const shaderModule = device.createShaderModule({ code: shader });
  #|    const bindGroupLayout = device.createBindGroupLayout({
  #|      entries: [
  #|        {
  #|          binding: 0,
  #|          visibility: GPUShaderStage.COMPUTE,
  #|          buffer: { type: "read-only-storage" },
  #|        },
  #|        {
  #|          binding: 1,
  #|          visibility: GPUShaderStage.COMPUTE,
  #|          buffer: { type: "read-only-storage" },
  #|        },
  #|        {
  #|          binding: 2,
  #|          visibility: GPUShaderStage.COMPUTE,
  #|          buffer: { type: "storage" },
  #|        },
  #|      ],
  #|    });
  #|    const computePipeline = device.createComputePipeline({
  #|      layout: device.createPipelineLayout({
  #|        bindGroupLayouts: [bindGroupLayout],
  #|      }),
  #|      compute: { module: shaderModule, entryPoint: "main" },
  #|    });
  #|    const bindGroup = device.createBindGroup({
  #|      layout: bindGroupLayout,
  #|      entries: [
  #|        { binding: 0, resource: { buffer: inputABuffer } },
  #|        { binding: 1, resource: { buffer: inputBBuffer } },
  #|        { binding: 2, resource: { buffer: outputBuffer } },
  #|      ],
  #|    });
  #|    const commandEncoder = device.createCommandEncoder();
  #|    const passEncoder = commandEncoder.beginComputePass();
  #|    passEncoder.setPipeline(computePipeline);
  #|    passEncoder.setBindGroup(0, bindGroup);
  #|    passEncoder.dispatchWorkgroups(Math.ceil(inputA.length / 64));
  #|    passEncoder.end();
  #|
  #|    commandEncoder.copyBufferToBuffer(
  #|      outputBuffer,
  #|      0,
  #|      stagingBuffer,
  #|      0,
  #|      length * 4
  #|    );
  #|
  #|    device.queue.submit([commandEncoder.finish()]);
  #|
  #|    await stagingBuffer.mapAsync(GPUMapMode.READ, 0, length * 4);
  #|
  #|    const output = new Float32Array(stagingBuffer.getMappedRange());
  #|    stagingBuffer.unmap();
  #|
  #|    while (true) {
  #|      if (Atomics.compareExchange(control, 0, 0, 1) === 0) {
  #|        for (let i = 0; i < length; i++) {
  #|          inputA[i] = output[i];
  #|        }
  #|        Atomics.store(control, 0, 2);
  #|        Atomics.notify(control, 0, 1);
  #|        break;
  #|      } else {
  #|        Atomics.wait(control, 0, 0);
  #|      }
  #|    }
  #|  };
  #|} else {
  #|  onMessage = (event) => {
  #|    const { buffer } = event;
  #|    const controlLength = 2;
  #|    const control = new Int32Array(buffer, 0, controlLength);
  #|    const length = control[1];
  #|    const inputA = new Float32Array(buffer, controlLength * 4, length);
  #|    const inputB = new Float32Array(
  #|      buffer,
  #|      controlLength * 4 + inputA.byteLength,
  #|      length
  #|    );
  #|    while (true) {
  #|      if (Atomics.compareExchange(control, 0, 0, 1) === 0) {
  #|        for (let i = 0; i < length; i++) {
  #|          inputA[i] = inputA[i] + inputB[i];
  #|        }
  #|        Atomics.store(control, 0, 2);
  #|        Atomics.notify(control, 0, 1);
  #|        break;
  #|      } else {
  #|        Atomics.wait(control, 0, 0);
  #|      }
  #|    }
  #|  };
  #|}
  #|
  #|if (isNode) {
  #|  const { parentPort } = await import("node:worker_threads");
  #|  if (!parentPort) {
  #|    throw Error("This script must be run as a worker");
  #|  }
  #|  parentPort.on("message", onMessage);
  #|} else {
  #|  self.onmessage = onMessage;
  #|}

///|
type Worker

///|
extern "js" fn Worker::new(source : String) -> Worker =
  #|(source) => {
  #|  let worker;
  #|  if (globalThis.Worker) {
  #|    const dataUrl = `data:text/javascript;base64,${btoa(source)}`;
  #|    worker = new Worker(dataUrl, { type: "module" });
  #|  } else {
  #|    const { Worker } = require("node:worker_threads");
  #|    worker = new Worker(source, { eval: true });
  #|  }
  #|  return worker;
  #|}

///|
extern "js" fn Worker::terminate(self : Worker) -> Unit =
  #|(worker) => {
  #|  worker.terminate();
  #|}

///|
type Float32Array

///|
extern "js" fn Float32Array::from_fixed(
  array : FixedArray[Float]
) -> Float32Array =
  #|(array) => new Float32Array(array)

///|
extern "js" fn Float32Array::to_fixed(self : Float32Array) -> FixedArray[Float] =
  #|(float32Array) => Array.from(float32Array)

///|
extern "js" fn worker_op_add_ffi(
  worker : Worker,
  a : Float32Array,
  b : Float32Array
) -> Float32Array =
  #|(worker, a, b) => {
  #|  const controlLength = 2;
  #|  const buffer = new SharedArrayBuffer(
  #|    controlLength * 4 + a.byteLength + b.byteLength
  #|  );
  #|  const control = new Int32Array(buffer, 0, controlLength);
  #|  control[0] = 0;
  #|  control[1] = a.length;
  #|  const inputA = new Float32Array(buffer, controlLength * 4, a.length);
  #|  inputA.set(a);
  #|  const inputB = new Float32Array(
  #|    buffer,
  #|    controlLength * 4 + a.byteLength,
  #|    b.length
  #|  );
  #|  inputB.set(b);
  #|  worker.postMessage({ buffer });
  #|  while (true) {
  #|    const state = Atomics.compareExchange(control, 0, 2, 3);
  #|    if (state === 2) {
  #|      return inputA;
  #|    } else if (state < 2) {
  #|      Atomics.wait(control, 0, state);
  #|    } else {
  #|      throw new Error("Invalid state");
  #|    }
  #|  }
  #|}

///|
fn worker_op_add(
  a : FixedArray[Float],
  b : FixedArray[Float]
) -> FixedArray[Float] {
  let worker = Worker::new(source)
  let a = Float32Array::from_fixed(a)
  let b = Float32Array::from_fixed(b)
  let result = worker_op_add_ffi(worker, a, b)
  worker.terminate()
  result.to_fixed()
}

///|
pub impl TensorAdd for Float with op_add(
  self : Tensor[Float],
  other : Tensor[Float]
) -> Tensor[Float] {
  let (self, other) = Tensor::broadcast(self, other)
  let value = worker_op_add(self.value._, other.value._)
  Tensor::{
    value,
    shape: self.shape,
    block: self.block,
    graph: Backward(GraphAdd((self, other))),
    refcnt: 0,
    grad: FixedArray::make(value.length(), (0.0 : Float)),
  }
}
