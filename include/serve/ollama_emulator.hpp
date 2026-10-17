#pragma once

#include "serve/c_api.h"
#include <string>
#include <atomic>
#include <thread>

namespace aie::serve {

class OllamaEmulator {
public:
    OllamaEmulator(aie_engine_t* engine, int port = 11434);
    ~OllamaEmulator();

    void start();
    void stop();

private:
    void listener_loop();
    void handle_client(uintptr_t client_socket);

    aie_engine_t* engine_;
    int port_;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    uintptr_t server_socket_{0};
};

} // namespace aie::serve
