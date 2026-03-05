'use strict';

import {
    IPCServer,
    IPCClient,
    IPCPorts,
    Tasks,
    Task,
    Log
} from '@src/classes';

export class SubProcess extends IPCServer {

    private client: IPCClient;
    private archiveTasksTimeout: ReturnType<typeof setTimeout>;

    constructor() {

        super(IPCPorts.subProcess());

        this.init();
        this.loopArchiveTasks();
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
    private async loopArchiveTasks() : Promise<void> {

        await this.archiveTasks();

        setTimeout(this.loopArchiveTasks.bind(this), 1000 * 60 * 60 * 24);
    }

    /*
    **
    **
    */
    private async archiveTasks() : Promise<void> {

        try {
            const tasks = new Tasks();

            const alivedTasksData = await tasks.getAlivedTasks();

            for (const data of alivedTasksData) {

                const task = new Task(data.id, data);

                if (task.data.status === 'DONE' && this.canArchive(new Date(task.data.time))) {
                
                    await task.setNewStatus('ARCHIVED');
                    
                    await this.client.send('/archived-task', task.id);

                    Log.log(`Task ${task.id} is now archived`);
                }
            }
        } catch(error) {
            console.log(error);
        }
    }

    /*
    **
    **
    */
    private canArchive(date: Date): boolean {
        
        const now = new Date();

        return now.getMonth() > date.getMonth() || now.getFullYear() > date.getFullYear();
    }

    /*
    **
    **
    */
    public release() : void {

        clearTimeout(this.archiveTasksTimeout);
    }
}