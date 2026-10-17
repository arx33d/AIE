#include "serve/ollama_emulator.hpp"
#include <iostream>
#include <cstring>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
#define CLOSE_SOCKET(s) closesocket(s)
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
using socket_t = int;
#define CLOSE_SOCKET(s) close(s)
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#endif

namespace aie::serve {

OllamaEmulator::OllamaEmulator(aie_engine_t* engine, int port)
    : engine_(engine), port_(port) {}

OllamaEmulator::~OllamaEmulator() {
    stop();
}

void OllamaEmulator::start() {
    running_.store(true);
    worker_thread_ = std::thread(&OllamaEmulator::listener_loop, this);
}

void OllamaEmulator::stop() {
    if (running_.load()) {
        running_.store(false);
        if (server_socket_ != 0 && server_socket_ != static_cast<uintptr_t>(INVALID_SOCKET)) {
            CLOSE_SOCKET(static_cast<socket_t>(server_socket_));
            server_socket_ = 0;
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
}

void OllamaEmulator::listener_loop() {
    socket_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == INVALID_SOCKET) return;

    int opt = 1;
#if defined(_WIN32)
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
        CLOSE_SOCKET(s);
        return;
    }

    if (listen(s, 16) == SOCKET_ERROR) {
        CLOSE_SOCKET(s);
        return;
    }

    server_socket_ = static_cast<uintptr_t>(s);
    std::cout << "[Ollama Wire Protocol] Listening on http://0.0.0.0:" << port_ << " (/api/chat, /api/tags)\n";

    while (running_.load()) {
        sockaddr_in client_addr;
#if defined(_WIN32)
        int len = sizeof(client_addr);
#else
        socklen_t len = sizeof(client_addr);
#endif
        socket_t client = accept(s, reinterpret_cast<sockaddr*>(&client_addr), &len);
        if (client == INVALID_SOCKET) break;

        std::thread(&OllamaEmulator::handle_client, this, static_cast<uintptr_t>(client)).detach();
    }
}

void OllamaEmulator::handle_client(uintptr_t client_socket) {
    socket_t s = static_cast<socket_t>(client_socket);
    char buf[4096];
    int recvd = recv(s, buf, sizeof(buf) - 1, 0);
    if (recvd <= 0) {
        CLOSE_SOCKET(s);
        return;
    }
    buf[recvd] = '\0';
    std::string req(buf);

    if (req.find("GET /api/tags") != std::string::npos) {
        std::string json = "{\"models\":[{\"name\":\"aie-model:latest\",\"model\":\"aie-model:latest\",\"details\":{\"format\":\"gguf\",\"family\":\"llama\"}}]}";
        std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                           std::to_string(json.size()) + "\r\n\r\n" + json;
        send(s, resp.c_str(), static_cast<int>(resp.size()), 0);
        CLOSE_SOCKET(s);
        return;
    }

    const char* chunked_hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/x-ndjson\r\n"
        "Connection: close\r\n\r\n";
    send(s, chunked_hdr, static_cast<int>(std::strlen(chunked_hdr)), 0);

    struct CbContext {
        socket_t sock;
    } ctx{s};

    aie_generation_params_t params;
    params.prompt = "Ollama prompt";
    params.max_tokens = 32;
    params.temperature = 0.7f;
    params.top_p = 0.9f;

    aie_generate_stream(
        engine_,
        &params,
        [](const char* token, int is_fin, void* udata) -> int {
            auto* c = reinterpret_cast<CbContext*>(udata);
            std::string line = "{\"model\":\"aie-model:latest\",\"message\":{\"role\":\"assistant\",\"content\":\"" +
                               std::string(token) + "\"},\"done\":" + (is_fin ? std::string("true") : std::string("false")) + "}\n";
            send(c->sock, line.c_str(), static_cast<int>(line.size()), 0);
            return 0;
        },
        &ctx
    );

    CLOSE_SOCKET(s);
}

} // namespace aie::serve
