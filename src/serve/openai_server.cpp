#include "serve/openai_server.hpp"
#include <iostream>
#include <sstream>
#include <vector>
#include <cstring>

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

OpenAIServer::OpenAIServer(aie_engine_t* engine, int port)
    : engine_(engine), port_(port) {
#if defined(_WIN32)
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

OpenAIServer::~OpenAIServer() {
    stop();
#if defined(_WIN32)
    WSACleanup();
#endif
}

void OpenAIServer::start() {
    running_.store(true);
    worker_thread_ = std::thread(&OpenAIServer::listener_loop, this);
}

void OpenAIServer::stop() {
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

void OpenAIServer::listener_loop() {
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
    std::cout << "[OpenAI Server] Listening on http://0.0.0.0:" << port_ << "/v1/chat/completions\n";

    while (running_.load()) {
        sockaddr_in client_addr;
#if defined(_WIN32)
        int len = sizeof(client_addr);
#else
        socklen_t len = sizeof(client_addr);
#endif
        socket_t client = accept(s, reinterpret_cast<sockaddr*>(&client_addr), &len);
        if (client == INVALID_SOCKET) break;

        std::thread(&OpenAIServer::handle_client, this, static_cast<uintptr_t>(client)).detach();
    }
}

void OpenAIServer::handle_client(uintptr_t client_socket) {
    socket_t s = static_cast<socket_t>(client_socket);
    char buf[4096];
    int recvd = recv(s, buf, sizeof(buf) - 1, 0);
    if (recvd <= 0) {
        CLOSE_SOCKET(s);
        return;
    }
    buf[recvd] = '\0';
    std::string req(buf);

    if (req.find("OPTIONS") == 0) {
        const char* cors =
            "HTTP/1.1 204 No Content\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n\r\n";
        send(s, cors, static_cast<int>(std::strlen(cors)), 0);
        CLOSE_SOCKET(s);
        return;
    }

    const char* sse_header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n\r\n";
    send(s, sse_header, static_cast<int>(std::strlen(sse_header)), 0);

    struct CbContext {
        socket_t sock;
    } ctx{s};

    aie_generation_params_t params;
    params.prompt = "User prompt";
    params.max_tokens = 32;
    params.temperature = 0.7f;
    params.top_p = 0.9f;

    aie_generate_stream(
        engine_,
        &params,
        [](const char* token, int is_fin, void* udata) -> int {
            auto* c = reinterpret_cast<CbContext*>(udata);
            std::string event = "data: {\"choices\":[{\"delta\":{\"content\":\"" + std::string(token) + "\"}}]}\n\n";
            send(c->sock, event.c_str(), static_cast<int>(event.length()), 0);

            if (is_fin) {
                const char* done = "data: [DONE]\n\n";
                send(c->sock, done, static_cast<int>(std::strlen(done)), 0);
            }
            return 0;
        },
        &ctx
    );

    CLOSE_SOCKET(s);
}

} // namespace aie::serve
