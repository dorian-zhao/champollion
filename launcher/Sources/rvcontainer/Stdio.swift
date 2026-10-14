import Containerization
import Foundation

/// The container process's output, written to one of our file handles.
final class FileWriter: Writer {
    let handle: FileHandle

    init(handle: FileHandle) {
        self.handle = handle
    }

    func write(_ data: Data) throws {
        try handle.write(contentsOf: data)
    }

    func close() throws {}
}

/// The container process's input, read from one of our file handles until end of file.
final class FileReader: ReaderStream {
    let handle: FileHandle

    init(handle: FileHandle) {
        self.handle = handle
    }

    func stream() -> AsyncStream<Data> {
        let handle = self.handle
        return AsyncStream { continuation in
            let thread = Thread {
                while true {
                    let data = handle.availableData
                    if data.isEmpty {
                        break
                    }
                    continuation.yield(data)
                }
                continuation.finish()
            }
            thread.start()
        }
    }
}
