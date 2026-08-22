// Unit test entry point (plan TD-2). Runs all Kit suites and exits non-zero on
// any failure: `swift run MeetingTranscriberKitTests`.

runServiceHandshakeTests()
runServiceDiscoveryTests()
runModelDecodingTests()
runAPIClientTests()
runProvisioningControllerTests()

TestRunner.shared.finish()
