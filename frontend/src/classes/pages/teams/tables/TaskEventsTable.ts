'use strict';

import {
    Table,
    Block, 
    Api
} from '@src/classes';

export class TaskEventsTable extends Table {

    constructor(private taskId: number, parent: Block) {

        super({
            title: "Task events",
            selectable: false
        }, parent);

        this.asyncPopulation();
    }
    
    /*
    **
    **
    */
    private async asyncPopulation() : Promise<void> {

        let rows: any[] = [];
        
        let entries: any[] = await Api.get('/task/events', {
            taskId: this.taskId
        });

        let columns = {
            'ID': Table.NUMBER,
            'Status': Table.STRING,
            'Time': Table.DATE
        }

        for (let entry of entries) {
            
            rows.push({
                'ID': entry.id,
                'Status': entry.status,
                'Time': new Date(entry.time).toLocaleString('en-US')
            });
        }

        this.populate(columns, rows);
    }
}