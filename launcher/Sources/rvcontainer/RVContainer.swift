import ArgumentParser
import Foundation

@main
struct RVContainer: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "rvcontainer",
        abstract: "Run riscv64 Linux container images through rvrun in a lightweight VM.",
        subcommands: [Run.self]
    )
}

/// Where rvcontainer keeps its state, and where it finds what it needs.
enum Paths {
    /// Images, the init file system, and per-container files.
    static let root: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("champollion")

    /// The default kernel, installed by `launcher/fetch-kernel.sh`.
    static var kernel: URL { root.appendingPathComponent("vmlinux") }

    /// The default Linux build of rvrun: `rvrun-linux` next to this executable, where
    /// `launcher/build.sh` puts it.
    static var rvrun: URL {
        URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            .deletingLastPathComponent().appendingPathComponent("rvrun-linux")
    }
}
