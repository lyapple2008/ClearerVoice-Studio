import AppKit
import AVKit
import Combine
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var model = StudioViewModel()

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Label("ClearerVoice Studio", systemImage: "waveform.and.magnifyingglass")
                    .font(.headline)

                if model.hasMedia {
                    Divider().frame(height: 18)
                    Label(
                        model.mediaFileName,
                        systemImage: model.hasVideo ? "film" : "waveform"
                    )
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                }

                Spacer()

                Button("导入音频或视频", systemImage: "square.and.arrow.down") {
                    presentImporter()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut("o", modifiers: .command)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)

            Divider()

            HStack(spacing: 0) {
                VStack(spacing: 16) {
                    VideoPreview(
                        player: model.videoPlayer,
                        hasMedia: model.hasMedia
                    )
                    .frame(maxHeight: .infinity)

                    AudioPreview(model: model)
                        .frame(height: 325)
                }
                .padding(20)
                .frame(minWidth: 720)

                Divider()

                FunctionMenu(model: model)
                    .frame(width: 320)
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .alert(
            "操作失败",
            isPresented: Binding(
                get: { model.errorMessage != nil },
                set: { if !$0 { model.errorMessage = nil } }
            )
        ) {
            Button("好", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "未知错误")
        }
    }

    private func presentImporter() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio, .movie, .video, .mpeg4Movie, .quickTimeMovie]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.prompt = "导入"
        panel.message = "选择要预览的音频或视频文件"

        guard panel.runModal() == .OK, let url = panel.url else { return }
        model.loadMedia(from: url)
    }
}

private struct VideoPreview: View {
    let player: AVPlayer?
    let hasMedia: Bool

    var body: some View {
        Panel(title: "视频预览", systemImage: "play.rectangle") {
            Group {
                if let player {
                    VideoPlayer(player: player)
                        .allowsHitTesting(false)
                } else {
                    EmptyMediaState(
                        systemImage: hasMedia ? "waveform" : "film.stack",
                        title: hasMedia ? "当前文件没有视频画面" : "尚未导入媒体",
                        subtitle: hasMedia
                            ? "音频可在下方的音频预览区播放"
                            : "使用顶部的导入按钮选择音频或视频文件"
                    )
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black.opacity(0.92))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        } trailing: {
            EmptyView()
        }
    }
}

private struct AudioPreview: View {
    @ObservedObject var model: StudioViewModel
    private let playbackClock = Timer.publish(every: 0.04, on: .main, in: .common).autoconnect()

    var body: some View {
        Panel(title: "音频预览", systemImage: "waveform") {
            if model.hasMedia {
                VStack(spacing: 10) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.mediaFileName)
                                .font(.subheadline.weight(.semibold))
                                .lineLimit(1)
                            Text("处理选区 \(formatTime(model.selectedDuration))")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text("\(formatTime(model.currentTime)) / \(formatTime(model.duration))")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    PCMSelectionWaveform(
                        samples: model.waveformSamples,
                        isLoading: model.isWaveformLoading,
                        duration: model.duration,
                        selectionStart: model.selectionStart,
                        selectionEnd: model.selectionEnd,
                        currentTime: model.currentTime,
                        onSeek: { model.seek(to: $0) },
                        onSelect: { model.setSelection(from: $0, to: $1) }
                    )
                    .frame(height: 96)

                    Text("单击波形跳转播放位置 · 拖动波形选择处理区间")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)

                    HStack(spacing: 12) {
                        Button {
                            model.togglePlayback()
                        } label: {
                            Label(
                                model.isPlaying ? "暂停" : "播放",
                                systemImage: model.isPlaying ? "pause.fill" : "play.fill"
                            )
                            .frame(width: 70)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.duration <= 0)

                        RangeValueControl(
                            title: "起点",
                            value: Binding(
                                get: { model.selectionStart },
                                set: { model.setSelectionStart($0) }
                            ),
                            range: 0...max(model.duration, 0.01)
                        )
                        RangeValueControl(
                            title: "终点",
                            value: Binding(
                                get: { model.selectionEnd },
                                set: { model.setSelectionEnd($0) }
                            ),
                            range: 0...max(model.duration, 0.01)
                        )
                    }
                }
            } else {
                EmptyMediaState(
                    systemImage: "waveform.path.badge.plus",
                    title: "尚未导入媒体",
                    subtitle: "导入后可在此切换播放位置并控制播放"
                )
            }
        } trailing: {
            if model.hasMedia {
                Label(model.hasVideo ? "视频音轨" : "音频文件", systemImage: model.hasVideo ? "film" : "music.note")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .onReceive(playbackClock) { _ in
            model.refreshPlaybackPosition()
        }
    }
}

private struct FunctionMenu: View {
    @ObservedObject var model: StudioViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("处理效果")
                    .font(.title2.bold())
                Text("选择要应用到音频选区的功能")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }

            VStack(spacing: 10) {
                ForEach(AudioEffect.allCases) { effect in
                    EffectButton(
                        effect: effect,
                        isSelected: model.selectedEffect == effect
                    ) {
                        model.selectedEffect = effect
                    }
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 8) {
                Label("当前选区", systemImage: "selection.pin.in.out")
                    .font(.headline)
                HStack {
                    Text(formatTime(model.selectionStart))
                    Image(systemName: "arrow.right")
                    Text(formatTime(model.selectionEnd))
                    Spacer()
                    Text(formatTime(model.selectedDuration))
                        .foregroundStyle(.secondary)
                }
                .font(.callout.monospacedDigit())
            }

            Spacer()

            if let receipt = model.lastReceipt {
                Label {
                    Text("已将“\(receipt.effect.title)”应用到 \(formatTime(receipt.startTime))–\(formatTime(receipt.endTime))")
                } icon: {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
                .font(.callout)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.green.opacity(0.09), in: RoundedRectangle(cornerRadius: 10))
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("ClearVoice 接口预留")
                    .font(.caption.weight(.semibold))
                Text(model.selectedEffect.clearVoiceModelType)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                Text("当前为占位处理器，不执行模型推理。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Button {
                model.applySelectedEffect()
            } label: {
                HStack {
                    if model.isProcessing {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: "wand.and.stars")
                    }
                    Text(model.isProcessing ? "正在应用…" : "应用到选区")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!model.hasMedia || model.isProcessing)
        }
        .padding(20)
        .background(Color(nsColor: .controlBackgroundColor))
    }
}

private struct PCMSelectionWaveform: View {
    let samples: [Float]
    let isLoading: Bool
    let duration: TimeInterval
    let selectionStart: TimeInterval
    let selectionEnd: TimeInterval
    let currentTime: TimeInterval
    let onSeek: (TimeInterval) -> Void
    let onSelect: (TimeInterval, TimeInterval) -> Void

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                Canvas { context, size in
                    drawWaveform(in: context, size: size)
                }
                .background(Color.black.opacity(0.78), in: RoundedRectangle(cornerRadius: 8))

                if isLoading {
                    ProgressView("正在读取 PCM…")
                        .controlSize(.small)
                        .foregroundStyle(.white)
                } else if samples.isEmpty {
                    Text("未检测到可用音轨")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.65))
                }
            }
            .background(Color.black.opacity(0.78), in: RoundedRectangle(cornerRadius: 8))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        if dragDistance(value) >= 4 {
                            onSelect(
                                time(at: value.startLocation.x, width: proxy.size.width),
                                time(at: value.location.x, width: proxy.size.width)
                            )
                        }
                    }
                    .onEnded { value in
                        if dragDistance(value) < 4 {
                            onSeek(time(at: value.location.x, width: proxy.size.width))
                        } else {
                            onSelect(
                                time(at: value.startLocation.x, width: proxy.size.width),
                                time(at: value.location.x, width: proxy.size.width)
                            )
                        }
                    }
            )
        }
    }

    private func drawWaveform(in context: GraphicsContext, size: CGSize) {
        guard !samples.isEmpty else { return }

        let selectionStartX = position(for: selectionStart, width: size.width)
        let selectionEndX = position(for: selectionEnd, width: size.width)
        let selectedRect = CGRect(
            x: selectionStartX,
            y: 1,
            width: max(1, selectionEndX - selectionStartX),
            height: max(1, size.height - 2)
        )
        context.fill(
            Path(roundedRect: selectedRect, cornerRadius: 5),
            with: .color(Color.accentColor.opacity(0.12))
        )

        let middleY = size.height / 2
        let columnWidth = size.width / CGFloat(samples.count)
        let lineWidth = max(1, min(3, columnWidth * 0.72))

        for (index, sample) in samples.enumerated() {
            let amplitude = max(0.015, min(1, CGFloat(sample)))
            let height = amplitude * max(1, size.height - 14)
            let x = (CGFloat(index) + 0.5) * columnWidth
            let sampleTime = duration * Double(index) / Double(max(1, samples.count - 1))
            let isSelected = sampleTime >= selectionStart && sampleTime <= selectionEnd
            var path = Path()
            path.move(to: CGPoint(x: x, y: middleY - height / 2))
            path.addLine(to: CGPoint(x: x, y: middleY + height / 2))
            context.stroke(
                path,
                with: .color(Color.accentColor.opacity(isSelected ? 0.95 : 0.32)),
                lineWidth: lineWidth
            )
        }

        context.stroke(
            Path(roundedRect: selectedRect, cornerRadius: 5),
            with: .color(Color.accentColor),
            lineWidth: 1.5
        )

        let playheadX = position(for: currentTime, width: size.width)
        var playhead = Path()
        playhead.move(to: CGPoint(x: playheadX, y: 0))
        playhead.addLine(to: CGPoint(x: playheadX, y: size.height))
        context.stroke(playhead, with: .color(.white), lineWidth: 1.5)
    }

    private func position(for time: TimeInterval, width: CGFloat) -> CGFloat {
        guard duration > 0 else { return 0 }
        return width * CGFloat(time / duration)
    }

    private func time(at x: CGFloat, width: CGFloat) -> TimeInterval {
        guard duration > 0, width > 0 else { return 0 }
        return duration * Double(max(0, min(width, x)) / width)
    }

    private func dragDistance(_ value: DragGesture.Value) -> CGFloat {
        hypot(value.translation.width, value.translation.height)
    }
}

private struct RangeValueControl: View {
    let title: String
    @Binding var value: TimeInterval
    let range: ClosedRange<TimeInterval>

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).foregroundStyle(.secondary)
                Spacer()
                Text(formatTime(value)).monospacedDigit()
            }
            .font(.caption)
            Slider(value: $value, in: range)
        }
    }
}

private struct EffectButton: View {
    let effect: AudioEffect
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: effect.systemImage)
                    .font(.title3)
                    .foregroundStyle(isSelected ? Color.white : Color.accentColor)
                    .frame(width: 28)
                VStack(alignment: .leading, spacing: 2) {
                    Text(effect.title).font(.headline)
                    Text(effect.subtitle)
                        .font(.caption)
                        .foregroundStyle(isSelected ? .white.opacity(0.78) : .secondary)
                        .lineLimit(2)
                }
                Spacer()
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                isSelected ? Color.accentColor : Color(nsColor: .windowBackgroundColor),
                in: RoundedRectangle(cornerRadius: 10)
            )
            .contentShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
    }
}

private struct Panel<Content: View, Trailing: View>: View {
    let title: String
    let systemImage: String
    @ViewBuilder let content: Content
    @ViewBuilder let trailing: Trailing

    init(
        title: String,
        systemImage: String,
        @ViewBuilder content: () -> Content,
        @ViewBuilder trailing: () -> Trailing
    ) {
        self.title = title
        self.systemImage = systemImage
        self.content = content()
        self.trailing = trailing()
    }

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Label(title, systemImage: systemImage)
                    .font(.headline)
                Spacer()
                trailing
            }
            content
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color(nsColor: .separatorColor).opacity(0.5))
        }
    }
}

private struct EmptyMediaState: View {
    let systemImage: String
    let title: String
    let subtitle: String

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: systemImage)
                .font(.system(size: 38))
                .foregroundStyle(.secondary)
            Text(title).font(.headline)
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private func formatTime(_ time: TimeInterval) -> String {
    guard time.isFinite, time >= 0 else { return "00:00.0" }
    let minutes = Int(time) / 60
    let seconds = time - Double(minutes * 60)
    return String(format: "%02d:%04.1f", minutes, seconds)
}
