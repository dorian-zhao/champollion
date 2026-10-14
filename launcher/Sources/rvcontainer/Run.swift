import ArgumentParser
import Containerization
import ContainerizationError
import ContainerizationExtras
import ContainerizationOCI
import ContainerizationOS
import Foundation

/// The init file system (vminitd) the VM boots from. Its agent protocol must match this
/// package's version of Containerization.
private let initImageReference = "ghcr.io/apple/containerization/vminit:0.48.0"

/// Where rvrun is shared into the VM (and the container) for binfmt_misc.
private let rvrunShare = "/run/champollion"

extension Binfmt.Entry {
    /// riscv64 ELF executables and shared objects, handled by rvrun. Flags: P passes the
    /// original argv[0] (rvrun reads AT_FLAGS), O and C let setuid programs run with their
    /// owner's credentials (rvrun reads the program from AT_EXECFD), and F opens rvrun at
    /// registration, so it is found from inside the container too.
    static func riscv64() -> Self {
        Binfmt.Entry(
            name: "riscv64",
            magic: #"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xf3\x00"#,
            mask: #"\xff\xff\xff\xff\xff\xff\xff\x00\xff\xff\xff\xff\xff\xff\xff\xff\xfe\xff\xff\xff"#,
            flags: "POCF"
        )
    }
}

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run a command in a new container from a riscv64 image.",
        discussion: "Pulls the image's linux/riscv64 variant if it is not stored yet."
    )

    @Option(help: "Linux kernel for the VM (default: ~/Library/Application Support/champollion/vmlinux).")
    var kernel: String?

    @Option(help: "Linux (aarch64) build of rvrun (default: rvrun-linux next to rvcontainer).")
    var rvrun: String?

    @Option(help: "Number of CPUs.")
    var cpus = 4

    @Option(help: "Memory in MiB.")
    var memory: UInt64 = 2048

    @Option(name: .customLong("fs-size"), help: "Size of the container's root file system in MiB.")
    var fsSize: UInt64 = 8192

    @Option(name: [.customShort("v"), .customLong("volume")], help: "Share a host directory: HOST:CONTAINER.")
    var volumes: [String] = []

    @Option(name: [.customShort("e"), .customLong("env")], help: "Set an environment variable: NAME=VALUE.")
    var environment: [String] = []

    @Option(name: [.customShort("w"), .customLong("workdir")], help: "Working directory in the container.")
    var workdir: String?

    @Flag(help: "Do not give the container a network.")
    var noNetwork = false

    @Argument(help: "Image reference, e.g. docker.io/library/debian:trixie.")
    var image: String

    @Argument(parsing: .captureForPassthrough, help: "Command and arguments (default: the image's).")
    var command: [String] = []

    func run() async throws {
        let kernelPath = URL(fileURLWithPath: kernel ?? Paths.kernel.path)
        guard FileManager.default.fileExists(atPath: kernelPath.path) else {
            throw ValidationError("no kernel at \(kernelPath.path): run launcher/fetch-kernel.sh, or pass --kernel")
        }
        let rvrunPath = URL(fileURLWithPath: rvrun ?? Paths.rvrun.path)
        guard FileManager.default.isExecutableFile(atPath: rvrunPath.path) else {
            throw ValidationError("no rvrun at \(rvrunPath.path): build it with launcher/build.sh, or pass --rvrun")
        }

        let store = try ImageStore(path: Paths.root.appendingPathComponent("images"))
        let vmm = try await Self.virtualMachineManager(store: store, kernel: kernelPath)

        let platform = Platform(arch: "riscv64", os: "linux")
        let image: Containerization.Image
        do {
            image = try await store.get(reference: self.image)
        } catch let error as ContainerizationError where error.code == .notFound {
            image = try await store.pull(reference: self.image, platform: platform)
        }
        let imageConfig = try await image.config(for: platform).config

        let id = "rv-" + String(UInt32.random(in: 0...UInt32.max), radix: 16)
        let containerDir = Paths.root.appendingPathComponent("containers").appendingPathComponent(id)
        try FileManager.default.createDirectory(at: containerDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: containerDir) }
        let rootfs = try await EXT4Unpacker(capacityInBytes: fsSize.mib())
            .unpack(image, for: platform, at: containerDir.appendingPathComponent("rootfs.ext4"))

        // rvrun goes in a directory of its own, the unit Virtualization.framework shares.
        let rvrunDir = containerDir.appendingPathComponent("rvrun")
        try FileManager.default.createDirectory(at: rvrunDir, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: rvrunPath, to: rvrunDir.appendingPathComponent("rvrun"))

        var network: VmnetNetwork? = noNetwork ? nil : try VmnetNetwork()
        defer { try? network?.releaseInterface(id) }

        let terminal = isatty(0) != 0 && isatty(1) != 0 ? try Terminal.current : nil
        var config = LinuxContainer.Configuration()
        if let imageConfig {
            config.process = .init(from: imageConfig)
        }
        if !command.isEmpty {
            // As with docker run: the arguments replace the image's CMD, not its ENTRYPOINT.
            config.process.arguments = (imageConfig?.entrypoint ?? []) + command
        }
        guard !config.process.arguments.isEmpty else {
            throw ValidationError("the image has no command; give one")
        }
        config.process.environmentVariables += environment
        if let workdir {
            config.process.workingDirectory = workdir
        }
        if let terminal {
            config.process.setTerminalIO(terminal: terminal)
        } else {
            config.process.stdin = FileReader(handle: .standardInput)
            config.process.stdout = FileWriter(handle: .standardOutput)
            config.process.stderr = FileWriter(handle: .standardError)
        }
        config.mounts.append(.share(source: rvrunDir.path, destination: rvrunShare))
        for volume in volumes {
            let parts = volume.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else {
                throw ValidationError("bad volume \(volume): expected HOST:CONTAINER")
            }
            let host = URL(fileURLWithPath: parts[0]).standardizedFileURL.path
            config.mounts.append(.share(source: host, destination: parts[1]))
        }
        if let interface = try network?.createInterface(id) {
            config.interfaces = [interface]
            if let gateway = interface.ipv4Gateway {
                config.dns = DNS(nameservers: [gateway.description])
            }
        }
        config.hosts = Hosts.default
        config.hostname = id

        let container = try LinuxContainer(
            id,
            rootfs: rootfs,
            vmm: vmm,
            vm: VMResources(cpus: cpus, memoryInBytes: memory.mib() + VMResources.guestMemoryOverhead),
            configuration: config
        )
        try await container.create()
        do {
            try await Self.registerRvrun(container: container, id: id)
            if let terminal { try terminal.setraw() }
            defer { terminal?.tryReset() }
            try await container.start()
            if let terminal { try? await container.resize(to: try terminal.size) }
            let status = try await container.wait()
            try await container.stop()
            if status.exitCode != 0 {
                terminal?.tryReset()
                throw ExitCode(status.exitCode)
            }
        } catch {
            try? await container.stop()
            throw error
        }
    }

    /// The VM manager, booting from the init image (pulled and converted on first use).
    static func virtualMachineManager(store: ImageStore, kernel: URL) async throws -> VZVirtualMachineManager {
        let initPath = Paths.root.appendingPathComponent("initfs-0.48.0.ext4")
        let initImage = try await store.getInitImage(reference: initImageReference)
        let initfs: Containerization.Mount
        do {
            initfs = try await initImage.initBlock(at: initPath, for: .linuxArm)
        } catch let error as ContainerizationError where error.code == .exists {
            initfs = .block(format: "ext4", source: initPath.path, destination: "/", options: ["ro"])
        }
        return VZVirtualMachineManager(kernel: Kernel(path: kernel, platform: .linuxArm), initialFilesystem: initfs)
    }

    /// Register rvrun, shared into the VM at `/run/virtiofs/<tag>`, as the handler of
    /// riscv64 programs. The registration belongs to the VM's kernel, so it covers the
    /// container's processes.
    static func registerRvrun(container: LinuxContainer, id: String) async throws {
        try await container.withVirtualMachineInstance { vm in
            guard let share = vm.mounts[id]?.first(where: { $0.destination == rvrunShare }) else {
                throw ContainerizationError(.notFound, message: "rvrun share not attached")
            }
            let agent = try await vm.dialAgent()
            defer { Task { try? await agent.close() } }
            guard let vminitd = agent as? Vminitd else {
                throw ContainerizationError(.unsupported, message: "the VM's agent cannot register binfmt_misc handlers")
            }
            try await vminitd.setupEmulator(
                binaryPath: "/run/virtiofs/\(share.source)/rvrun",
                configuration: .riscv64()
            )
        }
    }
}
