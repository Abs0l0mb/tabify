'use strict';

import {
    Table,
    TableRow, 
    Block, 
    Api,
    DeleteAccountSessionPopup
} from '@src/classes';

export class AccountSessionsTable extends Table {

    constructor(private accountId: number, parent: Block) {

        super({
            title: 'Sessions',
            rowOptions: [
                {
                    text: 'Delete',
                    event: 'delete'
                }
            ]
        }, parent);

        this.on('delete', (row: TableRow) => {
            new DeleteAccountSessionPopup(row.rowData.ID)
                .on('done', row.delete.bind(row));
        });

        this.asyncPopulation();
    }
    
    /*
    **
    **
    */
    private async asyncPopulation() : Promise<void> {

        let rows: any[] = [];
        
        let entries: any[] = await Api.get('/account/sessions', {
            id: this.accountId
        });


        let columns = {
            'ID': Table.NUMBER,
            'Create Date': Table.DATE,
            'Update Date': Table.DATE,
            //'Last Activity': Table.STRING,
            'Last IP': Table.STRING,
            'Browser Name': Table.STRING,
            'Browser Version': Table.STRING,
            'OS Name (UA)': Table.STRING,
            'OS Version (UA)': Table.STRING,
            'Device Type': Table.STRING
        }

        for (let entry of entries) {
            
            rows.push({
                'ID': entry.id,
                'Create Date': entry.create_date ? new Date(entry.create_date).toLocaleString('en-US') : null,
                'Update Date': entry.update_date ? new Date(entry.update_date).toLocaleString('en-US') : null,
                //'Last Activity': entry.last_activity,
                'Last IP': entry.last_ip,
                'Browser Name': entry.browser_name,
                'Browser Version': entry.browser_version,
                'OS Name (UA)': entry.os_name,
                'OS Version (UA)': entry.os_version,
                'Device Type': entry.device_type
            });
        }

        this.populate(columns, rows);
    }
}