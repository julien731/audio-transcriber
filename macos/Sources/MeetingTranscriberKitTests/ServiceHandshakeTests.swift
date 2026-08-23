import Foundation
import MeetingTranscriberKit

func runServiceHandshakeTests() {
    suite("ServiceHandshake.parse") {
        let ok = ServiceHandshake.parse(line: #"{"event":"ready","port":51234,"nonce":"abc"}"#)
        expectEqual(ok?.port, 51234, "parses port")
        expectEqual(ok?.nonce, "abc", "parses nonce")
        expectEqual(ok?.event, "ready", "event is ready")

        let noNonce = ServiceHandshake.parse(line: #"{"event":"ready","port":9}"#)
        expectEqual(noNonce?.port, 9, "nonce optional")
        expectNil(noNonce?.nonce, "missing nonce is nil")

        expectNil(ServiceHandshake.parse(line: "INFO: uvicorn running"), "ignores log lines")
        expectNil(ServiceHandshake.parse(line: #"{"event":"starting"}"#), "ignores non-ready events")
        expectNil(ServiceHandshake.parse(line: #"{"event":"ready"}"#), "requires a port")
        expectNil(ServiceHandshake.parse(line: "{not json"), "ignores malformed json")
        expectNil(ServiceHandshake.parse(line: ""), "ignores empty line")

        // Whitespace/mixed-output tolerance (a handshake line padded with spaces).
        let padded = ServiceHandshake.parse(line: #"   {"event":"ready","port":7}  "#)
        expectEqual(padded?.port, 7, "tolerates surrounding whitespace")
    }

    suite("ServiceHandshake.parseError") {
        let err = ServiceHandshake.parseError(line: #"{"event":"error","message":"no port"}"#)
        expectEqual(err, "no port", "parses error message")
        expectNil(ServiceHandshake.parseError(line: #"{"event":"ready","port":1}"#), "ready is not an error")
        expectNil(ServiceHandshake.parseError(line: "boom"), "ignores non-json")
    }
}
