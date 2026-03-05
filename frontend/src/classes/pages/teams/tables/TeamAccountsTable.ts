'use strict';

import {
    Table,
    TableRow, 
    Block, 
    Api,
    AddAccountPopup,
    EditAccountPopup,
    DeleteAccountPopup,
    Button
} from '@src/classes';

export class TeamAccountsTable extends Table {

    constructor(private teamId: number, readonly: boolean, parent: Block) {

        super({
            selectable: !readonly,
            rowOptions: !readonly ? [
                {
                    text: 'Edit',
                    event: 'edit'
                },
                {
                    text: 'Delete',
                    event: 'delete'
                }
            ] : [],
            actions: !readonly ? [
                {
                    text: 'Add account',
                    event: 'add'
                }
            ] : [],
        }, parent);

        if (!readonly) {
            this.on('edit', (row: TableRow) => {
                new EditAccountPopup(row.rowData.ID)
                    .on('success', () => this.asyncPopulation());
            });

            this.on('delete', (row: TableRow) => {
                new DeleteAccountPopup(row.rowData.ID)
                    .on('done', row.delete.bind(row));
            });

            this.on('add', (button: Button) => {
                button.load();
                let popup = new AddAccountPopup(this.teamId);
                popup.on('success', () => this.asyncPopulation());
                popup.on('hide', () => {
                    button.unload();
                });
            });
        }

        this.asyncPopulation();
    }
    
    /*
    **
    **
    */
    private async asyncPopulation() : Promise<void> {

        let rows: any[] = [];
        
        let entries: any[] = await Api.get('/team/accounts', {
            teamId: this.teamId
        });

        let columns = {
            'ID': Table.NUMBER,
            'Email': Table.STRING,
            'Last Name': Table.STRING,
            'First Name': Table.STRING,
            'Access Rights': Table.STRING
        }

        for (let entry of entries) {
            
            rows.push({
                'ID': entry.id,
                'Email': entry.email,
                'Last Name': entry.last_name,
                'First Name': entry.first_name,
                'Access Rights': entry.access_right_names ? entry.access_right_names.join(', ') : null,
            });
        }

        this.populate(columns, rows);
    }
}