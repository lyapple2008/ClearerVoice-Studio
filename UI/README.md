# ClearerVoice Studio for macOS

macOS 13+ SwiftUI 桌面应用。通过一个导入按钮选择音频或视频文件；视频画面只在视频预览区
显示，播放位置、播放/暂停和处理区间统一在音频预览区控制。视频画面与音频控制共享同一个
`AVPlayer` 播放时钟，并通过 `AudioProcessing` 协议预留 ClearVoice 接入点。

音频波形由媒体文件的真实 PCM 数据生成：纯音频通过 `AVAudioFile` 流式读取，视频音轨
通过 `AVAssetReaderAudioMixOutput` 解码。单击波形可跳转播放位置，拖动波形可选择处理区间。

当前的 `PlaceholderAudioProcessor` 不执行模型推理，仅用于验证完整 UI 交互。

## 使用 Xcode

1. 打开 `ClearerVoiceStudio.xcodeproj`。
2. 在 Target 的 **Signing & Capabilities** 中选择 Apple Developer Team。
3. 将 Bundle Identifier `com.clearervoice.studio` 改为你账户下的唯一标识。
4. Scheme 选择 **ClearerVoiceStudio > My Mac**，按 `Command-R` 运行。

工程已启用 App Sandbox 和 Hardened Runtime，并仅申请“用户选择文件只读”权限，
用于安全地预览用户主动导入的视频和音频。

## App Store 归档

1. 将运行目标选择为 **Any Mac (Apple Silicon, Intel)**。
2. 选择 **Product > Archive**。
3. 在 Organizer 中选择 **Distribute App > App Store Connect > Upload**。

上传前还需在 App Store Connect 中创建同 Bundle ID 的应用记录，填写隐私政策、截图、
分类和年龄分级等商店资料。代码签名证书、Provisioning Profile 和 App Store Connect 权限
由所选择的 Apple Developer Team 提供。

## Swift Package 运行

```bash
cd UI
swift run ClearerVoiceStudio
```

## 接入 ClearVoice

实现 `AudioProcessing` 协议，并在 `ContentView` 创建 `StudioViewModel` 时注入新的处理器。
请求中包含源文件 URL、选区起止时间和所选效果；效果同时提供对应的 ClearVoice 模型名称。
