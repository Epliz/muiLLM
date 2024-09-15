#include "comm_base.h"


#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <errno.h>
#include <poll.h>

#include <stdio.h>

#define MUILLM_COMM_SOCKET_PATH "/tmp/muillm_comm_sock13"

static int full_read(int fd, void* ptr, size_t byte_count) {
  size_t to_read_count = byte_count;

  uint8_t* byte_ptr = (uint8_t*) ptr;
  while (to_read_count > 0) {
    int read_status = read(fd, byte_ptr, to_read_count);

    if (read_status < 0) {
      return -1;
    }

    byte_ptr += read_status;
    to_read_count -= read_status;
  }

  return byte_count;
}

static int full_write(int fd, const void* ptr, size_t byte_count) {
  size_t to_write_count = byte_count;

  const uint8_t* byte_ptr = (const uint8_t*) ptr;
  while (to_write_count > 0) {
    int write_status = write(fd, byte_ptr, to_write_count);

    if (write_status < 0) {
      return -1;
    }

    byte_ptr += write_status;
    to_write_count -= write_status;
  }

  return byte_count;
}

// creates the domain sockets used to do cpu side exchanges
// (e.g. to exchange memory IPC handles)
muillm_comm_error_t __open_local_socket(
    muillm_comm_t* comm,
    int local_size,
    int local_rank
) {
  comm->server_fd = -1;
  comm->client_to_server_fd = -1;
  comm->server_to_client_fds = nullptr;

  bool is_server;

  // try to become the server
  struct sockaddr_un server_addr;

  // Create socket
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    return MUILLM_SOCKET_CREATION_FAILED;
  }

  // Try bindnig socket to address
  memset(&server_addr, 0, sizeof(struct sockaddr_un));
  server_addr.sun_family = AF_UNIX;
  strncpy(server_addr.sun_path, MUILLM_COMM_SOCKET_PATH, sizeof(server_addr.sun_path) - 1);

  // the server is always the rank 0
  is_server = local_rank == 0;

  if (is_server) {
    bool correctly_bound = bind(fd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_un)) == 0;

    if (!correctly_bound) {
      printf("(rank %d) Could not bind as server error was %s\n", local_rank, strerror(errno));
      return MUILLM_SOCKET_BIND_FAILED;
    }

    printf("(rank %d) I am the server\n", local_rank);
    // if we managed to bind, we should connect to all other ranks and they will be the socket
    // clients
    comm->server_fd = fd;

    int num_clients = local_size -1;

    comm->server_to_client_fds = new int[local_size];
    comm->server_to_client_fds[local_rank] = -1;

    // Listen for connections
    if (listen(comm->server_fd, num_clients) == -1) {
      printf("(rank %d) listen failed\n", local_rank);
      return MUILLM_SOCKET_LISTEN_FAILED;
    }

    struct pollfd poll_fd;
    poll_fd.fd = comm->server_fd;
    poll_fd.events = POLLIN;

    for (int c = 0; c < num_clients; c++) {
      printf("(rank %d) waiting for a client...\n", local_rank);

      int poll_count = poll(&poll_fd, 1, -1);
      if (poll_count < 0) {
        printf("(rank %d) Polling error, error was %s\n", local_rank, strerror(errno));
        return MUILLM_SOCKET_ACCEPT_FAILED;
      }

      if (poll_fd.revents & POLLIN) {
        int client_fd = accept(comm->server_fd, NULL, NULL);
        if (client_fd == -1) {
          printf("(rank %d) Accept failed error was %s\n", local_rank, strerror(errno));
          return MUILLM_SOCKET_ACCEPT_FAILED;
        }

        printf("(rank %d) new connection accepted\n", local_rank);
      
        // now read what rank this client corresponds to
        int client_rank = 0;

        if (full_read(client_fd, &client_rank, sizeof(int)) < 0) {
          printf("(rank %d) partial read\n", local_rank);
          return MUILLM_SOCKET_READ_ERROR;
        }

        printf("(rank %d) rank %d connected\n", local_rank, client_rank);
        comm->server_to_client_fds[client_rank] = client_fd;
      }
    }

    // we can already unlink as everyone has connected
    unlink(MUILLM_COMM_SOCKET_PATH);

  } else {
    printf("(rank %d) I am a client\n", local_rank);
    // if we didn't manage to bind, we are a mere client
    // we just need to connect to the server
    comm->client_to_server_fd = fd;

    bool connected = false;

    while (!connected) {
      if (connect(comm->client_to_server_fd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_un)) == -1) {
        printf("(rank %d) Could connect to the server error was %s\n", local_rank, strerror(errno));
        sleep(10); // wait before trying to connect again
        continue;
      }

      connected = true;
    }

    printf("(rank %d) connected to the server\n", local_rank);
    // send to the server our rank
    full_write(comm->client_to_server_fd, &local_rank, sizeof(int));
  }

  return MUILLM_COMM_SUCCESS;
}


// do a barrier using the local socker
void __local_socket_barrier(
    muillm_comm_t* comm
) {
  bool is_server = comm->server_fd != -1;
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  int client_to_server_val = 1;
  int server_to_client_val = 2;

  // the server waits for all other ranks to send a value
  // then sends all ranks something
    if (is_server) {
      // read from all the other ranks
      for (int r = 0; r < local_size; r++) {
        if (r == local_rank) continue;

        int v = 0;
        if (full_read(comm->server_to_client_fds[r], &v, sizeof(int)) < 0) {
          printf("partial write\n");
        }

        if (v != client_to_server_val) {
          printf("server read wrong barrier value\n");
        }
      }

      // then send something to all other ranks
      for (int r = 0; r < local_size; r++) {
        if (r == local_rank) continue;

        int v = server_to_client_val;
        if (full_write(comm->server_to_client_fds[r], &v, sizeof(int)) < 0) {
          printf("partial write\n");
        }
      }
    } else {
      // need to send to the server, the server will wait for all
      int v = client_to_server_val;
      if (full_write(comm->client_to_server_fd, &v, sizeof(int)) < 0) {
        printf("partial write\n");
      }

      // wait for the reply from the server
      if (full_read(comm->client_to_server_fd, &v, sizeof(int)) < 0) {
        printf("partial write\n");
      }

      if (v != server_to_client_val) {
          printf("client read wrong barrier value\n");
      }
    }
}

// do a broadcast using the local socker
void __local_socket_broadcast(
    muillm_comm_t* comm,
    int src_local_rank,
    void* ptr,
    size_t byte_count
) {
  bool is_server = comm->server_fd != -1;
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  if (src_local_rank == local_rank) {
    // we are the sender
    // we need to share the value with the other ranks
    if (is_server) {
      // just send to all other ranks
      for (int r = 0; r < local_size; r++) {
        if (r == local_rank) continue;
        if (full_write(comm->server_to_client_fds[r], ptr, byte_count) < 0) {
          printf("partial write\n");
        }
      }
    } else {
      // need to send to the server, the server will broadcast
      if (full_write(comm->client_to_server_fd, ptr, byte_count) < 0) {
        printf("partial write\n");
      }
    }
  } else {
    // we are a receiver
    // we need to receive the value from the src rank
    if (is_server) {
      // we need to receive it from the src rank first
      if (full_read(comm->server_to_client_fds[src_local_rank], ptr, byte_count) < 0) {
        printf("partial read\n");
      }
      // then broadcast to the others
      for (int r = 0; r < local_size; r++) {
        if (r == local_rank || r == src_local_rank) continue;
        if (full_write(comm->server_to_client_fds[r], ptr, byte_count) < 0) {
          printf("partial write\n");
        }
      }
    } else {
      // we need to receive from the server
      if (full_read(comm->client_to_server_fd, ptr, byte_count) < 0) {
        printf("partial read\n");
      }
    }
  }
}

// do a all gather using the local socket
// out_ptr is expected to have enough space for LOCAL_SIZE * byte_count
void __local_socket_all_gather(
    muillm_comm_t* comm,
    void* in_ptr,
    size_t byte_count,
    void* out_ptr
) {
  bool is_server = comm->server_fd != -1;
  int local_size = comm->local_size;
  int local_rank = comm->local_rank;

  if (is_server) {
    // receive from all ranks
    for (int r = 0; r < local_size; r++) {
      size_t offset = byte_count * r;
      if (r == local_rank) {
        // copy as well from ourself to put the content in in_ptr into out_ptr
        memcpy((int8_t*)out_ptr + offset, in_ptr, byte_count);
      } else {
        // copy the content from the remote rank into out_ptr
        if (full_read(comm->server_to_client_fds[r], (int8_t*)out_ptr + offset, byte_count) < 0) {
          printf("partial read\n");
        }
      }
    }

    // send to all other ranks
    for (int r = 0; r < local_size; r++) {
      if (r == local_rank) continue;
      if (full_write(comm->server_to_client_fds[r], out_ptr, byte_count * local_size) < 0) {
        printf("partial write\n");
      }
    }
  } else {
    // need to send to the server
    if (full_write(comm->client_to_server_fd, in_ptr, byte_count) < 0) {
      printf("partial write\n");
    }

    // need to received from the server
    if (full_read(comm->client_to_server_fd, out_ptr, byte_count * local_size) < 0) {
      printf("partial read\n");
    }
  }
}