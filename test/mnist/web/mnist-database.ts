/**
 * MNIST database.
 */
class MnistDatabase {
  private _database: IDBDatabase | IDBOpenDBRequest;
  private _dataset: Record<string, EventTarget | null>;
  /**
   * Create a new MNIST database.
   */
  constructor(name: string) {
    this._database = window.indexedDB.open(name);
    this._dataset = {
      test: null,
      train: null,
    };
  }
  /**
   * MNIST database URL.
   */
  private static readonly URL: Record<string, string> = {
    test: "https://media.githubusercontent.com/media/lorenmh/mnist_handwritten_json/master/mnist_handwritten_test.json.gz",
    train:
      "https://media.githubusercontent.com/media/lorenmh/mnist_handwritten_json/master/mnist_handwritten_train.json.gz",
  };
  /**
   * Fetch the MNIST data.
   */
  private static async fetchDataset(
    url: string,
  ): Promise<{ image: number[]; label: number }[]> {
    const response = await fetch(url);
    if (!response.ok || !response.body) {
      throw new Error(`Failed to fetch MNIST dataset from ${url}`);
    }
    const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
    /** @type {{ image: number[], label: number }[]} */
    return await new Response(stream).json();
  }
  /**
   * Upgrade the database.
   */
  private upgradeDatabase(database: IDBDatabase): void {
    for (const [name, url] of Object.entries(MnistDatabase.URL)) {
      if (this._dataset[name]) {
        continue;
      }
      const eventTarget = new EventTarget();
      this._dataset[name] = eventTarget;
      database.createObjectStore(name, {
        autoIncrement: true,
      });
      MnistDatabase.fetchDataset(url).then((dataset) => {
        const objectStore = database
          .transaction(name, "readwrite")
          .objectStore(name);
        for (const [_index, data] of dataset.entries()) {
          objectStore.add(data);
        }
        objectStore.transaction.oncomplete = () => {
          eventTarget.dispatchEvent(new CustomEvent("load"));
          this._dataset[name] = null;
        };
        objectStore.transaction.onerror = eventTarget.dispatchEvent;
      });
    }
  }
  /**
   * Get the database.
   */
  private async getDatabase(): Promise<IDBDatabase> {
    return await new Promise((resolve, reject) => {
      if (this._database instanceof IDBDatabase) {
        resolve(this._database);
      } else {
        this._database.addEventListener("success", (event) => {
          // @ts-ignore
          this._database = event.target.result;
          // @ts-ignore
          resolve(this._database);
        });
        this._database.addEventListener("error", (event) => {
          // @ts-ignore
          reject(event.target.error);
        });
        this._database.addEventListener("upgradeneeded", (event) => {
          // @ts-ignore
          this.upgradeDatabase(event.target.result);
        });
      }
    });
  }
  /**
   * Get the database.
   */
  async getDataset(
    name: string,
    mode: IDBTransactionMode,
  ): Promise<IDBObjectStore> {
    const database = await this.getDatabase();
    return new Promise((resolve) => {
      if (!this._dataset[name]) {
        resolve(database.transaction(name, mode).objectStore(name));
      } else {
        this._dataset[name].addEventListener("load", () => {
          resolve(database.transaction(name, mode).objectStore(name));
        });
      }
    });
  }
  /**
   * Get the data.
   */
  async getData(
    name: string,
    key: number,
  ): Promise<{ image: number[]; label: number }> {
    const objectStore = await this.getDataset(name, "readonly");
    const request = objectStore.get(key + 1);
    return new Promise((resolve, reject) => {
      request.onsuccess = (_event) => {
        // @ts-ignore
        resolve(request.result);
      };
      request.onerror = (event) => {
        // @ts-ignore
        reject(event.target?.error);
      };
    });
  }
}

const mnistDatabase = new MnistDatabase("mnist");

export default mnistDatabase;
