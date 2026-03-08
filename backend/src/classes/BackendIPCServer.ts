'use strict';

import {
    IPCServer,
    IPCPorts,
    IPCRequest,
    BackendHttpServer
} from "@src/classes";

export class BackendIPCServer extends IPCServer {
    
    constructor(private httpServer: BackendHttpServer) {

        super(IPCPorts.mainProcess());

        this.listen();
    }
}