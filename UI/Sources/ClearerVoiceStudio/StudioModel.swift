import AVFoundation
import Foundation

enum AudioEffect: String, CaseIterable, Identifiable, Sendable {
    case speechEnhancement
    case speechSeparation
    case superResolution
    case targetSpeakerExtraction

    var id: Self { self }

    var title: String {
        switch self {
        case .speechEnhancement: "语音增强"
        case .speechSeparation: "语音分离"
        case .superResolution: "语音超分"
        case .targetSpeakerExtraction: "目标说话人提取"
        }
    }

    var subtitle: String {
        switch self {
        case .speechEnhancement: "降低噪声并提升语音清晰度"
        case .speechSeparation: "从混合音频中分离不同说话人"
        case .superResolution: "补充高频细节并提升采样质量"
        case .targetSpeakerExtraction: "保留指定说话人的声音"
        }
    }

    var systemImage: String {
        switch self {
        case .speechEnhancement: "waveform.badge.plus"
        case .speechSeparation: "person.2.wave.2"
        case .superResolution: "arrow.up.right.and.arrow.down.left"
        case .targetSpeakerExtraction: "person.crop.circle.badge.checkmark"
        }
    }

    var clearVoiceModelType: String {
        switch self {
        case .speechEnhancement: "MossFormer2_SE_48K"
        case .speechSeparation: "MossFormer2_SS_16K"
        case .superResolution: "MossFormer2_SR_48K"
        case .targetSpeakerExtraction: "AV_MossFormer2_TSE_16K"
        }
    }
}

struct AudioProcessingRequest: Sendable {
    let sourceURL: URL
    let startTime: TimeInterval
    let endTime: TimeInterval
    let effect: AudioEffect
}

struct AudioProcessingReceipt: Sendable {
    let effect: AudioEffect
    let startTime: TimeInterval
    let endTime: TimeInterval
}

protocol AudioProcessing: Sendable {
    func process(_ request: AudioProcessingRequest) async throws -> AudioProcessingReceipt
}

/// Temporary implementation. Replace this type with a bridge to ClearVoice's C API.
struct PlaceholderAudioProcessor: AudioProcessing {
    func process(_ request: AudioProcessingRequest) async throws -> AudioProcessingReceipt {
        guard request.endTime > request.startTime else {
            throw AudioProcessingError.invalidRange
        }

        try await Task.sleep(for: .milliseconds(350))
        return AudioProcessingReceipt(
            effect: request.effect,
            startTime: request.startTime,
            endTime: request.endTime
        )
    }
}

enum AudioProcessingError: LocalizedError {
    case invalidRange

    var errorDescription: String? {
        switch self {
        case .invalidRange: "请选择有效的音频区间。"
        }
    }
}

@MainActor
final class StudioViewModel: ObservableObject {
    @Published private(set) var mediaURL: URL?
    @Published private(set) var mediaFileName = ""
    @Published private(set) var player: AVPlayer?
    @Published private(set) var hasVideo = false
    @Published private(set) var waveformSamples: [Float] = []
    @Published private(set) var isWaveformLoading = false
    @Published private(set) var duration: TimeInterval = 0
    @Published private(set) var currentTime: TimeInterval = 0
    @Published private(set) var isPlaying = false
    @Published var selectionStart: TimeInterval = 0
    @Published var selectionEnd: TimeInterval = 0
    @Published var selectedEffect: AudioEffect = .speechEnhancement
    @Published private(set) var isProcessing = false
    @Published private(set) var lastReceipt: AudioProcessingReceipt?
    @Published var errorMessage: String?

    private let processor: any AudioProcessing
    private let waveformSampler = WaveformSampler()
    private var scopedMediaURL: URL?
    private var durationLoadTask: Task<Void, Never>?
    private var waveformLoadTask: Task<Void, Never>?

    init(processor: any AudioProcessing = PlaceholderAudioProcessor()) {
        self.processor = processor
    }

    deinit {
        durationLoadTask?.cancel()
        waveformLoadTask?.cancel()
        scopedMediaURL?.stopAccessingSecurityScopedResource()
    }

    var hasMedia: Bool { mediaURL != nil }

    var videoPlayer: AVPlayer? { hasVideo ? player : nil }

    var selectedDuration: TimeInterval {
        max(0, selectionEnd - selectionStart)
    }

    func loadMedia(from url: URL) {
        pausePlayback()
        durationLoadTask?.cancel()
        waveformLoadTask?.cancel()
        releaseScopedAccess(for: scopedMediaURL)
        scopedMediaURL = beginScopedAccess(for: url)

        let asset = AVURLAsset(url: url)
        mediaURL = url
        mediaFileName = url.lastPathComponent
        hasVideo = false
        waveformSamples = []
        isWaveformLoading = true
        player = AVPlayer(playerItem: AVPlayerItem(asset: asset))
        duration = 0
        currentTime = 0
        selectionStart = 0
        selectionEnd = 0
        lastReceipt = nil
        errorMessage = nil

        durationLoadTask = Task { [weak self] in
            do {
                async let duration = asset.load(.duration)
                async let videoTracks = asset.loadTracks(withMediaType: .video)
                let (loadedDuration, loadedVideoTracks) = try await (duration, videoTracks)
                guard !Task.isCancelled,
                      let self,
                      self.mediaURL == url,
                      loadedDuration.seconds.isFinite,
                      loadedDuration.seconds > 0 else { return }

                self.duration = loadedDuration.seconds
                self.selectionEnd = loadedDuration.seconds
                self.hasVideo = !loadedVideoTracks.isEmpty
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled, let self, self.mediaURL == url else { return }
                self.errorMessage = "无法读取媒体文件：\(error.localizedDescription)"
            }
        }

        waveformLoadTask = Task { [weak self, waveformSampler] in
            do {
                let samples = try await waveformSampler.samples(from: url)
                guard !Task.isCancelled, let self, self.mediaURL == url else { return }
                self.waveformSamples = samples
                self.isWaveformLoading = false
            } catch is CancellationError {
                return
            } catch WaveformSamplingError.noAudioTrack {
                guard !Task.isCancelled, let self, self.mediaURL == url else { return }
                self.waveformSamples = []
                self.isWaveformLoading = false
            } catch {
                guard !Task.isCancelled, let self, self.mediaURL == url else { return }
                self.waveformSamples = []
                self.isWaveformLoading = false
                self.errorMessage = "无法生成音频波形：\(error.localizedDescription)"
            }
        }
    }

    func setSelectionStart(_ value: TimeInterval) {
        let minimumLength = min(0.1, duration)
        selectionStart = min(max(0, value), max(0, selectionEnd - minimumLength))
        lastReceipt = nil
    }

    func setSelectionEnd(_ value: TimeInterval) {
        let minimumLength = min(0.1, duration)
        selectionEnd = max(min(duration, value), min(duration, selectionStart + minimumLength))
        lastReceipt = nil
    }

    func setSelection(from firstTime: TimeInterval, to secondTime: TimeInterval) {
        let lowerBound = min(max(0, min(firstTime, secondTime)), duration)
        let upperBound = min(max(0, max(firstTime, secondTime)), duration)
        let minimumLength = min(0.1, duration)

        selectionStart = min(lowerBound, max(0, duration - minimumLength))
        selectionEnd = max(upperBound, min(duration, selectionStart + minimumLength))
        lastReceipt = nil
    }

    func togglePlayback() {
        guard let player, duration > 0 else { return }

        if player.timeControlStatus != .paused {
            pausePlayback()
            return
        }

        if currentTime >= duration - 0.05 {
            seek(to: 0)
        }
        player.play()
        isPlaying = true
    }

    func pausePlayback() {
        player?.pause()
        isPlaying = false
    }

    func seek(to value: TimeInterval) {
        guard let player else { return }

        let target = min(max(0, value), duration)
        currentTime = target
        player.seek(
            to: CMTime(seconds: target, preferredTimescale: 600),
            toleranceBefore: .zero,
            toleranceAfter: .zero
        )
    }

    func refreshPlaybackPosition() {
        guard let player else {
            isPlaying = false
            return
        }

        let playerTime = player.currentTime().seconds
        if playerTime.isFinite {
            currentTime = min(max(0, playerTime), duration)
        }
        isPlaying = player.timeControlStatus != .paused
    }

    func applySelectedEffect() {
        guard let mediaURL else { return }

        let request = AudioProcessingRequest(
            sourceURL: mediaURL,
            startTime: selectionStart,
            endTime: selectionEnd,
            effect: selectedEffect
        )
        isProcessing = true
        errorMessage = nil

        Task {
            do {
                lastReceipt = try await processor.process(request)
            } catch {
                errorMessage = error.localizedDescription
            }
            isProcessing = false
        }
    }

    private func beginScopedAccess(for url: URL) -> URL? {
        url.startAccessingSecurityScopedResource() ? url : nil
    }

    private func releaseScopedAccess(for url: URL?) {
        url?.stopAccessingSecurityScopedResource()
    }
}

enum WaveformSamplingError: LocalizedError {
    case noAudioTrack
    case invalidDuration
    case cannotAddOutput
    case readerFailed(String)

    var errorDescription: String? {
        switch self {
        case .noAudioTrack:
            "媒体文件不包含音轨。"
        case .invalidDuration:
            "无法读取媒体时长。"
        case .cannotAddOutput:
            "无法配置 PCM 解码输出。"
        case let .readerFailed(message):
            "PCM 解码失败：\(message)"
        }
    }
}

actor WaveformSampler {
    func samples(from url: URL, targetCount: Int = 600) async throws -> [Float] {
        do {
            return try samplesFromAudioFile(url, targetCount: targetCount)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return try await samplesFromAsset(url, targetCount: targetCount)
        }
    }

    private func samplesFromAudioFile(_ url: URL, targetCount: Int) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let totalFrames = file.length
        guard totalFrames > 0 else {
            throw WaveformSamplingError.invalidDuration
        }

        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat,
            frameCapacity: 8_192
        ) else {
            throw WaveformSamplingError.cannotAddOutput
        }

        var peaks = [Float](repeating: 0, count: targetCount)
        var processedFrames: AVAudioFramePosition = 0

        while processedFrames < totalFrames {
            if Task.isCancelled { throw CancellationError() }

            let remainingFrames = totalFrames - processedFrames
            let requestedFrames = AVAudioFrameCount(min(
                AVAudioFramePosition(buffer.frameCapacity),
                remainingFrames
            ))
            try file.read(into: buffer, frameCount: requestedFrames)
            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0, let channels = buffer.floatChannelData else { break }

            let channelCount = Int(buffer.format.channelCount)
            for frame in 0..<frameCount {
                var framePeak: Float = 0
                for channel in 0..<channelCount {
                    framePeak = max(framePeak, abs(channels[channel][frame]))
                }

                let bin = min(
                    targetCount - 1,
                    Int((processedFrames + AVAudioFramePosition(frame))
                        * AVAudioFramePosition(targetCount) / totalFrames)
                )
                peaks[bin] = max(peaks[bin], framePeak)
            }
            processedFrames += AVAudioFramePosition(frameCount)
        }

        return normalize(peaks)
    }

    private func samplesFromAsset(_ url: URL, targetCount: Int) async throws -> [Float] {
        let asset = AVURLAsset(url: url)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        guard let audioTrack = audioTracks.first else {
            throw WaveformSamplingError.noAudioTrack
        }

        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0 else {
            throw WaveformSamplingError.invalidDuration
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderAudioMixOutput(
            audioTracks: [audioTrack],
            audioSettings: [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false
            ]
        )
        output.alwaysCopiesSampleData = false

        guard reader.canAdd(output) else {
            throw WaveformSamplingError.cannotAddOutput
        }
        reader.add(output)
        guard reader.startReading() else {
            let errorDescription: String
            if let error = reader.error as NSError? {
                errorDescription = "\(error.domain) (\(error.code)): \(error.localizedDescription)"
            } else {
                errorDescription = "无法启动读取器"
            }
            throw WaveformSamplingError.readerFailed(
                errorDescription
            )
        }

        var peaks = [Float](repeating: 0, count: targetCount)
        var processedFrames = 0

        while let sampleBuffer = output.copyNextSampleBuffer() {
            if Task.isCancelled {
                reader.cancelReading()
                throw CancellationError()
            }

            guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer),
                  let streamDescription = CMAudioFormatDescriptionGetStreamBasicDescription(formatDescription) else {
                continue
            }

            let sampleRate = streamDescription.pointee.mSampleRate
            guard sampleRate > 0 else { continue }

            var bufferListSize = 0
            guard CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
                sampleBuffer,
                bufferListSizeNeededOut: &bufferListSize,
                bufferListOut: nil,
                bufferListSize: 0,
                blockBufferAllocator: nil,
                blockBufferMemoryAllocator: nil,
                flags: 0,
                blockBufferOut: nil
            ) == noErr,
            bufferListSize > 0 else { continue }

            let rawBufferList = UnsafeMutableRawPointer.allocate(
                byteCount: bufferListSize,
                alignment: MemoryLayout<AudioBufferList>.alignment
            )
            defer { rawBufferList.deallocate() }
            let bufferListPointer = rawBufferList.bindMemory(
                to: AudioBufferList.self,
                capacity: 1
            )
            var retainedBlockBuffer: CMBlockBuffer?
            let bufferStatus = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
                sampleBuffer,
                bufferListSizeNeededOut: nil,
                bufferListOut: bufferListPointer,
                bufferListSize: bufferListSize,
                blockBufferAllocator: kCFAllocatorDefault,
                blockBufferMemoryAllocator: kCFAllocatorDefault,
                flags: UInt32(kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment),
                blockBufferOut: &retainedBlockBuffer
            )
            guard bufferStatus == noErr else { continue }

            let audioBuffers = UnsafeMutableAudioBufferListPointer(bufferListPointer)
            guard let firstBuffer = audioBuffers.first else { continue }
            let firstChannelCount = max(1, Int(firstBuffer.mNumberChannels))
            let frameCount = Int(firstBuffer.mDataByteSize)
                / MemoryLayout<Int16>.size
                / firstChannelCount
            guard frameCount > 0 else { continue }

            let estimatedTotalFrames = max(1, Int(duration * sampleRate))

            for frame in 0..<frameCount {
                var framePeak: Float = 0
                for audioBuffer in audioBuffers {
                    guard let data = audioBuffer.mData else { continue }
                    let channelCount = max(1, Int(audioBuffer.mNumberChannels))
                    let pcm = data.assumingMemoryBound(to: Int16.self)
                    let frameOffset = frame * channelCount
                    for channel in 0..<channelCount {
                        let sample = abs(Int32(pcm[frameOffset + channel]))
                        framePeak = max(framePeak, Float(sample) / 32_768)
                    }
                }

                let bin = min(
                    targetCount - 1,
                    (processedFrames + frame) * targetCount / estimatedTotalFrames
                )
                peaks[bin] = max(peaks[bin], framePeak)
            }
            processedFrames += frameCount
        }

        if reader.status == .failed {
            throw WaveformSamplingError.readerFailed(
                reader.error?.localizedDescription ?? "未知错误"
            )
        }

        return normalize(peaks)
    }

    private func normalize(_ peaks: [Float]) -> [Float] {
        let maximumPeak = peaks.max() ?? 0
        guard maximumPeak > 0 else { return peaks }
        return peaks.map { min(1, $0 / maximumPeak) }
    }
}
