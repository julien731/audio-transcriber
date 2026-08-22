import AVFoundation
import Combine

/// Native audio playback (BR-8): streams the service's audio endpoint via AVPlayer
/// and publishes the current time so the transcript can highlight the active
/// segment. Replaces the web client's `<audio>` element while preserving synced
/// review, seek, and playback speed.
@MainActor
final class AudioPlaybackController: ObservableObject {
    @Published private(set) var currentTime: Double = 0
    @Published private(set) var duration: Double = 0
    @Published private(set) var isPlaying = false
    @Published var rate: Float = 1.0 {
        didSet { if isPlaying { player.rate = rate } }
    }

    private let player: AVPlayer
    private var timeObserver: Any?

    init(url: URL) {
        player = AVPlayer(url: url)
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 0.2, preferredTimescale: 600),
            queue: .main
        ) { [weak self] time in
            guard let self else { return }
            self.currentTime = time.seconds
            if let itemDuration = self.player.currentItem?.duration.seconds, itemDuration.isFinite {
                self.duration = itemDuration
            }
            self.isPlaying = self.player.timeControlStatus == .playing
        }
    }

    func toggle() {
        if isPlaying { pause() } else { play() }
    }

    func play() {
        player.play()
        player.rate = rate
        isPlaying = true
    }

    func pause() {
        player.pause()
        isPlaying = false
    }

    func seek(to seconds: Double) {
        let target = CMTime(seconds: max(0, seconds), preferredTimescale: 600)
        player.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
        currentTime = seconds
    }

    deinit {
        if let timeObserver { player.removeTimeObserver(timeObserver) }
    }
}
