'use strict';

import {
    IPCServer,
    IPCPorts,
    IPCRequest,
    BackendHttpServer,
    Task,
    Team
} from "@src/classes";

export class BackendIPCServer extends IPCServer {
    
    constructor(private httpServer: BackendHttpServer) {

        super(IPCPorts.mainProcess());

        this.on('/archived-task', this.onArchivedTask.bind(this));

        this.listen();
    }

    /*
    **
    **
    */
    private async onArchivedTask(request: IPCRequest) : Promise<void> {

        const taskId = request.getData();

        const task = new Task(taskId);

        await task.load();

        if (!task.data)
            return;

        this.httpServer.tasksWebSocketChannel.broadcastTaskSyncToTeam(new Team(task.data.team_id), task);
    }
}