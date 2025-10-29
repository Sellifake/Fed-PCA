"""
Trainers Module for Fed-ProFiLA-AD
"""

from .client_loop import Client, ClientManager
from .server_loop import Server, ServerManager

__all__ = ['Client', 'ClientManager', 'Server', 'ServerManager']
