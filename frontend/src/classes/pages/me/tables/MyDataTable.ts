'use strict';

import {
    Table,
    TableRow,
    Block, 
    Api,
    EditMyDataPopup
} from '@src/classes';

export class MyDataTable extends Table {

    constructor(parent: Block) {

        super({
            rowOptions: [
                {
                    text: 'Edit',
                    event: 'edit'
                }
            ]
        }, parent);

        this.on('edit', (tableRow: TableRow) => {
            new EditMyDataPopup()
                .on('success', tableRow.update.bind(tableRow));
        });

        this.asyncPopulation();
    }
    
    /*
    **
    **
    */
    private async asyncPopulation() : Promise<void> {

        let rows: any[] = [];
        
        let data: any = await Api.get('/me');

        let columns = {
            'ID': Table.NUMBER,
            'Email': Table.STRING,
            'Last Name': Table.STRING,
            'First Name': Table.STRING,
            'Access Rights': Table.STRING
        }

        rows.push({
            'ID': data.id,
            'Email': data.email,
            'Last Name': data.last_name,
            'First Name': data.first_name,
            'Access Rights': data.access_right_names ? data.access_right_names.join(', ') : null
        });

        this.populate(columns, rows);
    }
}