import process from "node:process";
import * as model from "../web/mnist-model";
import { gunzipSync } from "node:zlib";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const usage = `${process.argv[1]} <command> [options]`;

type Data = {
  image: number[];
  label: number;
};

class MnistMain {
  urls = [
    "https://media.githubusercontent.com/media/lorenmh/mnist_handwritten_json/master/mnist_handwritten_test.json.gz",
    "https://media.githubusercontent.com/media/lorenmh/mnist_handwritten_json/master/mnist_handwritten_train.json.gz",
  ];
  checksums = {
    "mnist_handwritten_test.json.gz":
      "bb6ea3726e5109a5564ee18ee1dc3695b87309e59181d47d8b620047260293b4",
    "mnist_handwritten_train.json.gz":
      "773a299b5c3d057baab1c6d32d4344796b9d1ee33f561ddbb5683b4fb61cd189",
  };

  private calculateChecksum(filePath: string): string {
    const buffer = fs.readFileSync(filePath);
    const hash = crypto.createHash("sha256");
    hash.update(buffer);
    return hash.digest("hex");
  }

  async fetch(dataDir: string) {
    for (const url of this.urls) {
      const fileName = url.split("/").pop();
      if (!fileName) {
        throw new Error(`Invalid URL: ${url}`);
      }
      const filePath = path.join(dataDir, fileName);
      const expectedChecksum = this.checksums[fileName];

      if (fs.existsSync(filePath)) {
        const actualChecksum = this.calculateChecksum(filePath);
        if (actualChecksum === expectedChecksum) {
          console.log(
            `File ${fileName} already exists and has correct checksum.`
          );
          continue;
        } else {
          console.log(`Checksum mismatch for ${fileName}`);
        }
      }

      console.log(`Fetching ${fileName}`);
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to fetch ${url}`);
      }
      if (!response.body) {
        throw new Error(`Failed to fetch ${url}`);
      }
      const buffer = await response.bytes();
      fs.writeFileSync(filePath, buffer);
      console.log(`Saved ${fileName}`);
    }
  }
  private readDataset(path: string): Data[] {
    const buffer = fs.readFileSync(path);
    const content = gunzipSync(buffer);
    const textDecoder = new TextDecoder();
    return JSON.parse(textDecoder.decode(content)) as Data[];
  }
  train(
    dataDir: string,
    options: {
      epoch: number;
      batchSize: number;
      learningRate: number;
    }
  ) {
    const batchSize = options.batchSize;
    const filePath = path.join(dataDir, "mnist_handwritten_train.json.gz");
    const dataset = this.readDataset(filePath);
    for (let epochIndex = 0; epochIndex < options.epoch; epochIndex++) {
      for (
        let dataIndex = 0;
        dataIndex < dataset.length;
        dataIndex += batchSize
      ) {
        const batch: [Float32Array, number][] = [];
        for (const data of dataset.slice(dataIndex, dataIndex + batchSize)) {
          batch.push([
            new Float32Array(data.image.map((value) => value / 255.0)),
            data.label,
          ]);
        }
        const loss = model.train(batch, options.learningRate);
        console.log("loss", loss);
      }
    }
  }

  infer(dataDir: string) {
    const filePath = path.join(dataDir, "mnist_handwritten_test.json.gz");
    const dataset = this.readDataset(filePath);
    let correct = 0;
    for (let index = 0; index < dataset.length; index++) {
      const image = new Float32Array(
        dataset[index].image.map((value) => value / 255.0)
      );
      const logits = model.infer(image);
      const prediction = logits.indexOf(Math.max(...logits));
      if (prediction === dataset[index].label) {
        correct++;
      }
    }
    const accuracy = correct / dataset.length;
    console.log("accuracy", accuracy);
  }

  load(path: string) {
    const content = new TextDecoder().decode(fs.readFileSync(path));
    model.load(content);
  }
  save(path: string) {
    const content = model.save();
    fs.writeFileSync(path, new TextEncoder().encode(content));
  }
}

type Command = "fetch" | "train" | "infer";

function validateCommand(command: string): Command {
  if (command === "fetch" || command === "train" || command === "infer") {
    return command;
  } else {
    throw new Error(`Invalid command: ${command}`);
  }
}

class ArgumentParser {
  private indexSet: Set<number>;
  constructor() {
    this.indexSet = new Set();
    for (let i = 3; i < process.argv.length; i++) {
      this.indexSet.add(i);
    }
  }
  option<T>(name: string, type: (string: string) => T) {
    let index = -1;
    if (index === -1) {
      index = process.argv.lastIndexOf(name);
    }
    if (index !== -1) {
      this.indexSet.delete(index);
      const value = process.argv[index + 1];
      if (!value || value.startsWith("-")) {
        throw new Error(`Error: expecting value after ${name}`);
      }
      this.indexSet.delete(index + 1);
      return type(value);
    }
  }
  value(): string | undefined {
    const posIndex = Array.from(this.indexSet).sort();
    if (posIndex.length === 0) {
      return;
    }
    this.indexSet.delete(posIndex[0]);
    return process.argv[posIndex[0]];
  }
}

const main = () => {
  const command = process.argv[2];
  if (command === undefined) {
    console.error("Error: expecting an command: {fetch,train,infer}");
    console.log(usage);
    process.exit(1);
  }
  const parser = new ArgumentParser();
  const main = new MnistMain();
  switch (validateCommand(command)) {
    case "fetch":
      {
        const dataDir = process.argv[3];
        if (!dataDir) {
          console.error("Error: <data-dir> is required for fetch");
          process.exit(1);
        }
        main.fetch(dataDir);
      }
      break;
    case "infer":
      {
        const dataDir = parser.option("--data", String) ?? "data";
        const modelPath = parser.value();
        if (!modelPath) {
          console.error("Error: <model> is required for infer");
          process.exit(1);
        }
        main.load(modelPath);
        main.infer(dataDir);
      }
      break;
    case "train": {
      const options = {
        epoch: parser.option("--epoch", Number) ?? 3,
        batchSize: parser.option("--batch-size", Number) ?? 256,
        learningRate: parser.option("--learning-rate", Number) ?? 0.01,
      };
      const dataDir = parser.option("--data", String) ?? "data";
      let outputPath =
        parser.option("--output", String) ?? parser.option("-o", String);
      const modelPath = parser.value();
      if (modelPath) {
        main.load(modelPath);
        if (!outputPath) {
          outputPath = modelPath;
        }
      }
      main.train(dataDir, options);
      if (outputPath) {
        main.save(outputPath);
      }
    }
  }
};

main();
