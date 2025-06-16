import SwiftUI
import AVKit
import UniformTypeIdentifiers // 1. 引入UTType，用于指定文件类型

struct VideoSource: Identifiable, Hashable {
    let id = UUID()
    let name: String
    let url: URL
}

struct ContentView: View {

    // 2. 默认视频列表保持为静态，方便初始化
    static private let defaultVideoSources: [VideoSource] = [
        VideoSource(name: "Nature", url: URL(string: "https://devstreaming-cdn.apple.com/videos/streaming/examples/bipbop_16x9/bipbop_16x9_variant.m3u8")!),
        VideoSource(name: "Fractals", url: URL(string: "https://bitdash-a.akamaihd.net/content/sintel/hls/playlist.m3u8")!),
        VideoSource(name: "Big Buck Bunny", url: URL(string: "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4")!)
    ]

    // 3. 将视频列表和播放器设为 @State，以便动态修改
    @State private var videoSources: [VideoSource]
    @State private var selectedVideoSource: VideoSource
    @State private var player: AVPlayer

    // 4. 新增一个 @State 变量来控制文件选择器的显示
    @State private var isFileImporterPresented = false

    init() {
        // 从默认列表初始化 State
        let initialSources = Self.defaultVideoSources
        let initialSource = initialSources[0]
        
        _videoSources = State(initialValue: initialSources)
        _selectedVideoSource = State(initialValue: initialSource)
        _player = State(initialValue: AVPlayer(url: initialSource.url))
    }

    var body: some View {
        VideoPlayer(player: player)
            .onAppear {
                player.play()
            }
            .onDisappear {
                player.pause()
            }
            .edgesIgnoringSafeArea(.all)
            .ornament(
                visibility: .automatic,
                attachmentAnchor: .scene(.bottom)
            ) {
                // 5.  使用全新的、更清晰的控制栏
                HStack(spacing: 12) {
                    // 自定义的视频切换按钮
                    ForEach(videoSources) { source in
                        Button(action: {
                            selectedVideoSource = source
                        }) {
                            // 给网络视频和本地视频不同的图标
                            let isLocal = source.url.isFileURL
                            Label(source.name, systemImage: isLocal ? "folder.fill" : "globe")
                        }
                        // 给当前选中的按钮一个高亮效果
                        .buttonStyle(.bordered)
                        .tint(selectedVideoSource == source ? .accentColor : .secondary)
                    }

                    // 增加一个分割线，区分功能区
                    Divider().padding(.horizontal, 4)

                    // 6.  增加“从本地选择”的按钮
                    Button(action: {
                        isFileImporterPresented = true // 触发文件选择器
                    }) {
                        Label("本地视频", systemImage: "plus")
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding()
                .glassBackgroundEffect()
            }
            // 7.  绑定文件选择器
            .fileImporter(
                isPresented: $isFileImporterPresented,
                allowedContentTypes: [UTType.movie, UTType.video], // 只允许选择电影/视频文件
                allowsMultipleSelection: false
            ) { result in
                handleFileImport(result: result)
            }
            .onChange(of: selectedVideoSource) { _, newSource in
                let newPlayerItem = AVPlayerItem(url: newSource.url)
                player.replaceCurrentItem(with: newPlayerItem)
                player.play()
            }
            .frame(minWidth: 1280, minHeight: 720)
    }

    // 8.  处理文件导入的逻辑
    private func handleFileImport(result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            
            // 访问沙盒环境外的文件需要获取安全访问权限
            let isAccessing = url.startAccessingSecurityScopedResource()
            guard isAccessing else {
                print("无法获取文件访问权限: \(url.lastPathComponent)")
                return
            }
            
            // 用文件名创建一个新的 VideoSource
            let newSourceName = url.deletingPathExtension().lastPathComponent
            let newSource = VideoSource(name: newSourceName, url: url)
            
            // 添加到列表并立即播放
            videoSources.append(newSource)
            selectedVideoSource = newSource

            // 注意：理想情况下，你需要在不再需要文件时调用 url.stopAccessingSecurityScopedResource()
            // 例如在 App 生命周期结束或手动移除该视频时。
            
        case .failure(let error):
            print("文件选择失败: \(error.localizedDescription)")
        }
    }
}
