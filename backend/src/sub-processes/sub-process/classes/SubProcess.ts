'use strict';

import {
    IPCServer,
    IPCClient,
    IPCPorts,
    Log
} from '@src/classes';

export class SubProcess extends IPCServer {

    private client: IPCClient;
    private archiveTasksTimeout: ReturnType<typeof setTimeout>;

    constructor() {

        super(IPCPorts.subProcess());

        this.init();
    }

    /*
    **
    **
    */
    private async init() : Promise<void> {

        await this.listen();

        if (process.send)
            process.send('ready');

        this.client = new IPCClient(IPCPorts.mainProcess());

        this.client.connect();
    }

    /*
    **
    **
    */
    public release() : void {

        clearTimeout(this.archiveTasksTimeout);
    }
}