// swift-tools-version: 6.2
// rvcontainer: run riscv64 Linux container images on a Mac (VM mode). Each container gets a
// lightweight arm64 Linux VM (apple/containerization on Virtualization.framework) in which
// rvrun, built for Linux, is registered with binfmt_misc for riscv64 programs.

import PackageDescription

let package = Package(
    name: "rvcontainer",
    platforms: [.macOS("26.0")],
    dependencies: [
        .package(url: "https://github.com/apple/containerization.git", exact: "0.48.0"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.7.0"),
    ],
    targets: [
        .executableTarget(
            name: "rvcontainer",
            dependencies: [
                .product(name: "Containerization", package: "containerization"),
                .product(name: "ContainerizationOCI", package: "containerization"),
                .product(name: "ContainerizationOS", package: "containerization"),
                .product(name: "ContainerizationExtras", package: "containerization"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        )
    ]
)
