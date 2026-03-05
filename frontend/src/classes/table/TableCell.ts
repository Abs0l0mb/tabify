'use strict';

import { 
    Block,
    Table
} from '@src/classes';

export class TableCell extends Block {
    
    constructor(public value: any, private columnType: number, parent: Block) {

        super('td', 'data-cell', parent);

        this.setValue(value);
    }

    /*
    **
    **
    */
    public setValue(value: any) : void {

        if (this.columnType === Table.DATE) {

            this.write(value ? new Date(value).toLocaleString('fr') : 'null');
            this.value = new Date(value).toLocaleString('fr');
        }
        else {
            
            this.write(value);
            this.value = value;
        }

        if (parseFloat(value) === 0.0 
         || !value
         || value === 'null'
         || value === '')
            this.setData('priority', -1);
        else
            this.setData('priority', 0);
    }
}