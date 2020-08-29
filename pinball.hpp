#include <algorithm>
#include <fcntl.h>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <termios.h>
#include <thread>
#include <unistd.h>

/// tty represents a generic serial connection.
/// timeout is a number of 0.1 s intervals.
class tty {
    public:
    tty(const std::string& filename, uint64_t baudrate, uint64_t timeout) :
        _filename(filename), _file_descriptor(open(_filename.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK)) {
        {
            auto flags = fcntl(_file_descriptor, F_GETFL);
            flags &= ~O_NONBLOCK;
            fcntl(_file_descriptor, F_SETFL, flags);
        }
        if (_file_descriptor < 0) {
            throw std::runtime_error(std::string("opening '") + _filename + "' failed");
        }
        termios options;
        if (tcgetattr(_file_descriptor, &options) < 0) {
            throw std::logic_error("getting the terminal options failed");
        }
        cfmakeraw(&options);
        cfsetispeed(&options, baudrate);
        cfsetospeed(&options, baudrate);
        options.c_cflag |= CREAD | CLOCAL;
        options.c_cc[VMIN] = 0;
        options.c_cc[VTIME] = timeout;
        tcsetattr(_file_descriptor, TCSANOW, &options);
        if (tcsetattr(_file_descriptor, TCSAFLUSH, &options) < 0) {
            throw std::logic_error("setting the terminal options failed");
        }
        tcflush(_file_descriptor, TCIOFLUSH);
    }
    tty(const tty&) = delete;
    tty(tty&&) = delete;
    tty& operator=(const tty&) = delete;
    tty& operator=(tty&&) = delete;
    virtual ~tty() {
        close(_file_descriptor);
    }

    /// write sends data to the tty.
    template <typename Byte>
    void write(const std::vector<Byte>& bytes) {
        if (::write(_file_descriptor, reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()) != bytes.size()) {
            throw std::runtime_error("write error");
        }
        tcdrain(_file_descriptor);
    }

    /// read loads a single byte from the tty.
    uint8_t read() {
        uint8_t byte;
        const auto bytes_read = ::read(_file_descriptor, &byte, 1);
        if (bytes_read <= 0) {
            if (access(_filename.c_str(), F_OK) < 0) {
                throw std::logic_error(std::string("'") + _filename + "' disconnected");
            }
            throw std::runtime_error("read timeout");
        }
        return byte;
    }

    protected:
    const std::string _filename;
    int32_t _file_descriptor;
};

/// pinball implements the communication with the Pinball machine.
class pinball {
    public:
    /// message encodes pinball commands.
    enum class message : uint8_t {
        launch = '2',
        kick_left = '4',
        kick_right = '6',
        hold_left = '1',
        hold_right = '3',
        release_left = '7',
        release_right = '9',
    };

    pinball(const std::string& filename) : _tty(filename, B230400, 1), _running(true), _ball_in_launcher(false) {
        for (;;) {
            try {
                const auto reset = _tty.read();
                if (reset == 'r') {
                    break;
                } else {
                    _tty.write<uint8_t>({0xff});
                }
            } catch (const std::runtime_error&) {
                continue;
            }
        }
        _loop = std::thread([this]() {
            while (_running.load(std::memory_order_relaxed)) {
                uint8_t byte;
                try {
                    byte = _tty.read();
                } catch (const std::runtime_error&) {
                    continue;
                }
                switch (byte) {
                    case 'r':
                        _ball_in_launcher.store(false, std::memory_order_release);
                        break;
                    case 'b':
                        _ball_in_launcher.store(true, std::memory_order_release);
                        break;
                    case 'l':
                        _ball_in_launcher.store(false, std::memory_order_release);
                        break;
                    default:
                        break;
                }
            }
        });
    }
    pinball(const pinball&) = delete;
    pinball(pinball&&) = delete;
    pinball& operator=(const pinball&) = delete;
    pinball& operator=(pinball&&) = delete;
    virtual ~pinball() {
        _running.store(false, std::memory_order_relaxed);
        _loop.join();
    }

    /// ball_in_launcher returns true if a ball is ready to be launched.
    bool ball_in_launcher() {
        return _ball_in_launcher.load(std::memory_order_acquire);
    }

    /// send gives an order to the pinball machine.
    void send(const std::vector<message>& messages) {
        if (std::any_of(messages.begin(), messages.end(), [](message value) { return value == message::launch; })) {
            _ball_in_launcher.store(false, std::memory_order_release);
        }
        _tty.write(messages);
    }
    void send(message value) {
        send(std::vector<message>{value});
    }

    protected:
    tty _tty;
    std::thread _loop;
    std::atomic_bool _running;
    std::string _message;
    std::atomic_bool _ball_in_launcher;
};
